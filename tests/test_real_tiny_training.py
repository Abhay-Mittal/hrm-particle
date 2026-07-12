from __future__ import annotations

import pytest
import torch

from hrm_particle.adapter import ParticleAdapterConfig
from hrm_particle.model import ParticleHrmForCausalLM
from hrm_particle.rollout import (
    ParticleRolloutEngine,
    RolloutConfig,
    RolloutExample,
    score_actions,
)
from hrm_particle.trainer import ParticleTrainer, TrainerConfig


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [2 + (ord(character) % 250) for character in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(
            chr((int(token) - 2) % 250)
            for token in ids
            if int(token) > 1 or not skip_special_tokens
        )


def _fixed_latents(batch_size, k, latent_dim, **kwargs):
    latents = torch.zeros(
        batch_size,
        k,
        latent_dim,
        device=kwargs["device"],
        dtype=kwargs["dtype"],
    )
    # Each nonzero row has unit RMS for latent_dim=4, while particle zero is the
    # exact clean anchor required by the rollout engine.
    latents[:, 1, 0] = 2.0
    latents[:, 2, 0] = -2.0
    latents[:, 3, 1] = 2.0
    return latents


@pytest.mark.integration
def test_real_tiny_hrm_cached_rollout_replay_and_finite_training_step() -> None:
    """Regress real-hook/autograd failures that a contract-only dummy cannot see."""

    try:
        from transformers import HrmTextConfig, HrmTextForCausalLM
    except ImportError:
        pytest.skip("installed Transformers version does not include HRM-Text")

    torch.manual_seed(20260709)
    config = HrmTextConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=8,
        max_position_embeddings=128,
        H_cycles=2,
        L_cycles=1,
        L_bp_cycles=[1, 1],
        num_layers_per_stack=1,
        prefix_lm=True,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
    )
    config._attn_implementation = "eager"
    model = ParticleHrmForCausalLM(
        HrmTextForCausalLM(config),
        adapter_config=ParticleAdapterConfig(
            hidden_size=32,
            latent_size=4,
            bottleneck_size=8,
            initial_relative_rms=0.03,
            max_relative_rms=0.10,
        ),
        q_bottleneck_size=8,
    )
    engine = ParticleRolloutEngine(
        model,
        _TinyTokenizer(),
        config=RolloutConfig(
            k=4,
            latent_dim=4,
            max_new_tokens=2,
            temperature=0.8,
            top_p=1.0,
            response_prefix="S",
            first_token_mode="causal_prefix",
            use_cache=True,
            compute_reference_logprobs=True,
            seed=7,
        ),
        latent_sampler=_fixed_latents,
    )

    rollout = engine.generate([RolloutExample("Compute one plus one.", "2", "tiny")])
    replay = score_actions(model, rollout)
    replay_mask = rollout.generated_mask
    assert torch.allclose(
        replay.logprobs[replay_mask],
        rollout.old_logprobs[replay_mask],
        atol=1e-6,
        rtol=1e-6,
    )
    ratios = torch.exp(replay.logprobs[replay_mask] - rollout.old_logprobs[replay_mask])
    assert torch.allclose(ratios, torch.ones_like(ratios), atol=1e-6, rtol=1e-6)

    # Supply deterministic external verifier labels so the actor and Q both
    # receive a nonzero learning signal independent of random tiny-model text.
    rollout.rewards.copy_(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    actor_optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-3)
    q_optimizer = torch.optim.AdamW(model.q_head.parameters(), lr=1e-3)
    trainer = ParticleTrainer(
        model,
        engine,
        actor_optimizer,
        q_optimizer,
        config=TrainerConfig(
            kl_coefficient=0.02,
            injection_penalty_coefficient=1e-3,
        ),
    )
    adapter_before = [parameter.detach().clone() for parameter in model.adapter.parameters()]
    q_before = [parameter.detach().clone() for parameter in model.q_head.parameters()]

    metrics = trainer.train_rollout(rollout)

    assert metrics["actor/mean_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["actor/grad_norm"] > 0.0
    assert metrics["q/grad_norm"] > 0.0
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert all(torch.isfinite(parameter).all() for parameter in model.adapter.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.q_head.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.adapter.parameters()
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(adapter_before, model.adapter.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(q_before, model.q_head.parameters())
    )
