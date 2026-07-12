from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hrm_particle.adapter import ParticleAdapterConfig
from hrm_particle.model import (
    HrmIntegrationError,
    ParticleHrmForCausalLM,
    validate_hrm_text_model,
)


class TinyCache:
    def __init__(self) -> None:
        self.seen_tokens = 0


class TinyStack(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, **_kwargs) -> torch.Tensor:
        return self.norm(hidden_states + torch.tanh(self.proj(hidden_states)))


class TinyHrmCore(nn.Module):
    def __init__(self, vocab_size: int = 23, hidden_size: int = 12, high_cycles: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            H_cycles=high_cycles,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.L_module = TinyStack(hidden_size)
        self.H_module = TinyStack(hidden_size)
        self.gradient_checkpointing = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        high = self.embed_tokens(input_ids)
        low = torch.zeros_like(high)
        for _ in range(self.config.H_cycles):
            low = self.L_module(low + high)
            high = self.H_module(high + low)
        return high


class TinyHrmForCausalLM(nn.Module):
    """Pointwise fake HRM: cached and full forwards are exactly comparable."""

    def __init__(self, high_cycles: int = 2) -> None:
        super().__init__()
        self.model = TinyHrmCore(high_cycles=high_cycles)
        self.config = self.model.config
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.raise_after_recurrence = False
        self.high_calls_override: int | None = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        past_key_values: TinyCache | None = None,
        use_cache: bool | None = None,
        return_dict: bool = True,
        **_kwargs,
    ) -> SimpleNamespace:
        del attention_mask, token_type_ids, return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one input source")
        high = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        low = torch.zeros_like(high)
        cycles = (
            self.config.H_cycles
            if self.high_calls_override is None
            else self.high_calls_override
        )
        for _ in range(cycles):
            low = self.model.L_module(low + high)
            high = self.model.H_module(high + low)
        if self.raise_after_recurrence:
            raise RuntimeError("intentional fake-model failure")
        if use_cache:
            past_key_values = past_key_values or TinyCache()
            past_key_values.seen_tokens += high.shape[1]
        return SimpleNamespace(
            logits=self.lm_head(high),
            past_key_values=past_key_values,
            loss=None,
            hidden_states=None,
            attentions=None,
            last_hidden_state=high,
        )


def make_wrapper(base: TinyHrmForCausalLM | None = None) -> ParticleHrmForCausalLM:
    base = base or TinyHrmForCausalLM()
    config = ParticleAdapterConfig(
        hidden_size=base.config.hidden_size,
        latent_size=4,
        bottleneck_size=8,
        max_relative_rms=0.10,
        initial_relative_rms=0.03,
    )
    return ParticleHrmForCausalLM(base, adapter_config=config, q_bottleneck_size=8)


def test_capability_validation_finds_hf_style_core() -> None:
    base = TinyHrmForCausalLM()
    capabilities = validate_hrm_text_model(base)

    assert capabilities.core_path == "base_model.model"
    assert capabilities.hidden_size == 12
    assert capabilities.high_cycles == 2
    assert capabilities.high_module is base.model.H_module


def test_zero_particle_wrapper_is_logit_exact() -> None:
    base = TinyHrmForCausalLM()
    wrapper = make_wrapper(base)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    query = torch.randn(2, 12)

    reference = base(input_ids=input_ids).logits
    output = wrapper(
        input_ids=input_ids,
        particle_z=torch.zeros(2, 4),
        particle_mask=mask,
        query_state=query,
    )

    assert torch.equal(output.logits, reference)
    assert output.injection_delta is not None
    assert torch.count_nonzero(output.injection_delta) == 0


def test_particle_mask_preserves_prompt_logits_exactly() -> None:
    base = TinyHrmForCausalLM()
    wrapper = make_wrapper(base)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    response_mask = torch.tensor([[False, False, True, True]])
    query = torch.randn(1, 12)
    z = torch.randn(1, 4)

    reference = base(input_ids=input_ids).logits
    output = wrapper(
        input_ids=input_ids,
        particle_z=z,
        particle_mask=response_mask,
        query_state=query,
    )

    assert torch.equal(output.logits[:, :2], reference[:, :2])
    assert not torch.equal(output.logits[:, 2:], reference[:, 2:])


def test_clean_prompt_summary_selects_final_prefix_state() -> None:
    wrapper = make_wrapper()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.tensor([[1, 1, 0, 0]])

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    expected = output.base_output.last_hidden_state[:, 1].detach()
    assert torch.equal(output.prompt_summary, expected)
    assert not torch.equal(output.prompt_summary, output.base_output.last_hidden_state[:, -1])


def test_explicit_prompt_mask_takes_precedence() -> None:
    wrapper = make_wrapper()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_type_ids = torch.tensor([[1, 1, 1, 0]])
    prompt_mask = torch.tensor([[1, 1, 0, 0]])

    output = wrapper(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        prompt_mask=prompt_mask,
    )

    assert torch.equal(output.prompt_summary, output.base_output.last_hidden_state[:, 1])


def test_terminal_state_matches_state_that_produces_last_logit() -> None:
    wrapper = make_wrapper()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    attention_mask = torch.ones_like(input_ids)
    terminal_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]])

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        terminal_mask=terminal_mask,
    )

    positions = torch.tensor([3, 2])
    expected_logits = wrapper.base_model.lm_head(output.terminal_state)
    actual_logits = output.logits[torch.arange(2), positions]
    assert torch.allclose(expected_logits, actual_logits)
    assert output.q_logits is not None and output.q_logits.shape == (2,)


def test_full_and_incremental_particle_forwards_match_on_tiny_model() -> None:
    wrapper = make_wrapper()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    full_mask = torch.tensor([[0, 0, 1, 1]])
    query = torch.randn(1, 12)
    z = torch.randn(1, 4)

    full = wrapper(
        input_ids=input_ids,
        particle_z=z,
        particle_mask=full_mask,
        query_state=query,
        use_cache=False,
    )

    cache = None
    step_logits = []
    for position in range(input_ids.shape[1]):
        is_response = position >= 2
        step_kwargs = {}
        if is_response:
            step_kwargs.update(
                particle_z=z,
                particle_mask=torch.ones(1, 1, dtype=torch.bool),
                query_state=query,
            )
        step = wrapper(
            input_ids=input_ids[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
            **step_kwargs,
        )
        cache = step.past_key_values
        step_logits.append(step.logits)

    incremental_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(full.logits, incremental_logits, atol=1e-6, rtol=1e-6)
    assert cache.seen_tokens == input_ids.shape[1]


def test_full_prompt_final_position_can_branch_first_response_logit() -> None:
    base = TinyHrmForCausalLM()
    wrapper = make_wrapper(base)
    prompt = torch.tensor([[1, 2, 3]])
    clean = wrapper(input_ids=prompt, prompt_mask=torch.ones_like(prompt))
    final_prompt_mask = torch.tensor([[0, 0, 1]])

    branch = wrapper(
        input_ids=prompt,
        particle_z=torch.randn(1, 4),
        particle_mask=final_prompt_mask,
        query_state=clean.prompt_summary,
    )
    reference = base(input_ids=prompt).logits

    assert not torch.equal(branch.logits[:, -1], reference[:, -1])
    assert torch.equal(branch.logits[:, :-1], reference[:, :-1])


def test_hook_and_lock_are_cleaned_up_after_base_exception() -> None:
    base = TinyHrmForCausalLM()
    wrapper = make_wrapper(base)
    hooks_before = len(base.model.H_module._forward_hooks)
    base.raise_after_recurrence = True

    with pytest.raises(RuntimeError, match="intentional"):
        wrapper(input_ids=torch.tensor([[1, 2]]))

    assert len(base.model.H_module._forward_hooks) == hooks_before
    base.raise_after_recurrence = False
    # A second call proves that the non-reentrant lock was also released.
    assert wrapper(input_ids=torch.tensor([[1, 2]])).logits.shape == (1, 2, 23)


def test_hook_is_cleaned_up_after_invocation_count_error() -> None:
    base = TinyHrmForCausalLM()
    wrapper = make_wrapper(base)
    hooks_before = len(base.model.H_module._forward_hooks)
    base.high_calls_override = 1

    with pytest.raises(HrmIntegrationError, match="invocation count"):
        wrapper(input_ids=torch.tensor([[1, 2]]))

    assert len(base.model.H_module._forward_hooks) == hooks_before


def test_gradient_checkpointing_fails_loudly() -> None:
    wrapper = make_wrapper()
    wrapper.base_model.model.gradient_checkpointing = True

    with pytest.raises(HrmIntegrationError, match="gradient checkpointing"):
        wrapper(input_ids=torch.tensor([[1, 2]]))


def test_particle_requires_explicit_mask_and_clean_query_state() -> None:
    wrapper = make_wrapper()
    z = torch.randn(1, 4)

    with pytest.raises(ValueError, match="particle_mask"):
        wrapper(input_ids=torch.tensor([[1, 2]]), particle_z=z)
    with pytest.raises(ValueError, match="query_state"):
        wrapper(
            input_ids=torch.tensor([[1, 2]]),
            particle_z=z,
            particle_mask=torch.ones(1, 2),
        )


def test_invalid_architecture_and_cycle_are_rejected() -> None:
    with pytest.raises(HrmIntegrationError, match="could not find"):
        validate_hrm_text_model(nn.Linear(2, 2))
    with pytest.raises(HrmIntegrationError, match="downstream high cycle"):
        validate_hrm_text_model(TinyHrmForCausalLM(), injection_after_high_cycle=1)
    with pytest.raises(HrmIntegrationError, match="H_cycles"):
        validate_hrm_text_model(TinyHrmForCausalLM(high_cycles=1))


@pytest.mark.integration
def test_official_tiny_hrm_zero_anchor_and_cached_prefixlm_parity() -> None:
    """Exercise the hook against the real offline Transformers implementation."""

    try:
        from transformers import HrmTextConfig, HrmTextForCausalLM
    except ImportError:
        pytest.skip("installed Transformers version does not include HRM-Text")

    torch.manual_seed(20260709)
    config = HrmTextConfig(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=8,
        max_position_embeddings=32,
        H_cycles=2,
        L_cycles=1,
        L_bp_cycles=[1, 1],
        num_layers_per_stack=1,
        prefix_lm=True,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    # Eager attention makes this a strict recurrence/cache test rather than a
    # comparison between two sequence-length-dependent fused kernels.
    config._attn_implementation = "eager"
    base = HrmTextForCausalLM(config).eval()
    wrapper = ParticleHrmForCausalLM(
        base,
        adapter_config=ParticleAdapterConfig(
            hidden_size=32,
            latent_size=4,
            bottleneck_size=8,
            max_relative_rms=0.10,
            initial_relative_rms=0.03,
        ),
        q_bottleneck_size=8,
    )

    prompt_ids = torch.tensor([[5, 6, 7]])
    full_ids = torch.tensor([[5, 6, 7, 8, 9]])
    full_attention = torch.ones_like(full_ids)
    full_token_types = torch.tensor([[1, 1, 1, 0, 0]])
    full_particle_mask = torch.tensor([[0, 0, 0, 1, 1]])
    particle_z = torch.tensor([[0.2, -0.4, 0.6, 0.8]])

    with torch.no_grad():
        clean = wrapper(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            token_type_ids=torch.ones_like(prompt_ids),
            use_cache=True,
        )
        query = clean.prompt_summary
        assert query is not None

        reference_logits = base(
            input_ids=full_ids,
            attention_mask=full_attention,
            token_type_ids=full_token_types,
            use_cache=False,
        ).logits
        zero_logits = wrapper(
            input_ids=full_ids,
            attention_mask=full_attention,
            token_type_ids=full_token_types,
            particle_z=torch.zeros_like(particle_z),
            particle_mask=full_particle_mask,
            query_state=query,
            use_cache=False,
        ).logits
        assert torch.equal(zero_logits, reference_logits)

        full_particle = wrapper(
            input_ids=full_ids,
            attention_mask=full_attention,
            token_type_ids=full_token_types,
            particle_z=particle_z,
            particle_mask=full_particle_mask,
            query_state=query,
            use_cache=False,
        )

        cache = clean.past_key_values
        incremental_logits = []
        for position, token in enumerate((8, 9), start=3):
            step = wrapper(
                input_ids=torch.tensor([[token]]),
                attention_mask=torch.ones(1, position + 1, dtype=torch.long),
                token_type_ids=torch.zeros(1, 1, dtype=torch.long),
                position_ids=torch.tensor([[position]]),
                past_key_values=cache,
                use_cache=True,
                particle_z=particle_z,
                particle_mask=torch.ones(1, 1, dtype=torch.long),
                query_state=query,
            )
            cache = step.past_key_values
            incremental_logits.append(step.logits)

    cached_particle_logits = torch.cat(incremental_logits, dim=1)
    assert torch.allclose(
        full_particle.logits[:, 3:], cached_particle_logits, atol=1e-6, rtol=1e-6
    )
