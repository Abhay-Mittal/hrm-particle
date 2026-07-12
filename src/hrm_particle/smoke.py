"""Offline end-to-end dummy hook used by ``scripts/run_smoke.py``."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .rollout import ParticleRolloutEngine, RolloutConfig, RolloutExample
from .trainer import ParticleTrainer, TrainerConfig


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 4 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            chr(token - 4)
            for token in ids
            if token >= 4 and not (skip_special_tokens and token in {0, 1})
        )


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.weight[0, 0] = 2.0
        self.embedding = nn.Embedding(260, 4)
        self.embedding.weight.requires_grad_(False)
        self.prefix_id = ord("§") + 4
        self.one_id = ord("1") + 4
        self.two_id = ord("2") + 4

    def forward(
        self,
        input_ids,
        attention_mask,
        particle_z=None,
        particle_mask=None,
        prompt_summary=None,
        query_state=None,
        **_kwargs,
    ):
        hidden = self.embedding(input_ids)
        batch, sequence = input_ids.shape
        query = query_state if query_state is not None else prompt_summary
        positions = torch.arange(sequence, device=input_ids.device).expand(batch, -1)
        last = positions.masked_fill(~attention_mask.bool(), -1).max(dim=-1).values
        if query is None:
            query = hidden[torch.arange(batch), last].detach()
        logits = hidden.new_full((batch, sequence, 260), -12.0)
        logits[..., 1] = 8.0
        delta = hidden.new_zeros(hidden.shape)
        relative = hidden.new_zeros((batch, sequence, 1))
        steering = hidden.new_zeros(batch)
        if particle_z is not None:
            steering = self.adapter(particle_z).squeeze(-1)
            mask = particle_mask[:, -sequence:].bool()
            delta = mask[..., None] * steering[:, None, None]
            relative = mask[..., None] * steering[:, None, None].abs() / 10.0
            hidden = hidden + delta
        rows, columns = input_ids.eq(self.prefix_id).nonzero(as_tuple=True)
        if rows.numel():
            logits[rows, columns, 1] = -12.0
            logits[rows, columns, self.one_id] = 1.0 - steering[rows]
            logits[rows, columns, self.two_id] = steering[rows]
        terminal = hidden[torch.arange(batch), last]
        return SimpleNamespace(
            logits=logits,
            past_key_values=None,
            terminal_state=terminal,
            prompt_summary=query,
            injection_delta=delta if particle_z is not None else None,
            relative_rms=relative if particle_z is not None else None,
        )


class _Q(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 1)

    def forward(self, terminal_state, prompt_summary):
        return self.projection(torch.cat((terminal_state, prompt_summary), dim=-1)).squeeze(-1)


def _latents(batch_size, k, latent_dim, **kwargs):
    value = torch.zeros(
        batch_size, k, latent_dim, device=kwargs["device"], dtype=kwargs["dtype"]
    )
    value[:, 1, 0] = 1.0
    value[:, 2:, 0] = -1.0
    return value


def run_dummy_smoke() -> dict[str, object]:
    """Exercise prompt masks, rollout, exact replay, PPO, and detached Q on CPU."""

    torch.manual_seed(20260709)
    model = _Policy()
    q_head = _Q()
    engine = ParticleRolloutEngine(
        model,
        _Tokenizer(),
        config=RolloutConfig(
            k=4,
            latent_dim=2,
            max_new_tokens=2,
            temperature=1.0,
            top_p=1.0,
            response_prefix="§",
            first_token_mode="causal_prefix",
            use_cache=False,
            compute_reference_logprobs=True,
            seed=11,
        ),
        latent_sampler=_latents,
    )
    actor_optimizer = torch.optim.Adam(model.adapter.parameters(), lr=0.05)
    q_optimizer = torch.optim.Adam(q_head.parameters(), lr=0.01)
    trainer = ParticleTrainer(
        model,
        engine,
        actor_optimizer,
        q_optimizer,
        q_head=q_head,
        config=TrainerConfig(
            kl_coefficient=0.0,
            injection_penalty_coefficient=0.0,
        ),
    )
    before = model.adapter.weight.detach().clone()
    result = trainer.train_batch(
        [RolloutExample(prompt="What is one plus one?", answer="2", example_id="offline")]
    )
    if result.rollout.rewards.tolist() != [[0.0, 1.0, 0.0, 0.0]]:
        raise AssertionError(f"unexpected dummy verifier rewards: {result.rollout.rewards}")
    if torch.equal(before, model.adapter.weight):
        raise AssertionError("dummy actor update did not change the adapter")
    return {
        "status": "passed",
        "anchor_zero_exact": bool((result.rollout.particle_z[:, 0] == 0).all()),
        "rewards": result.rollout.rewards.tolist(),
        "metrics": result.metrics,
    }


__all__ = ["run_dummy_smoke"]
