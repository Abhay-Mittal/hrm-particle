#!/usr/bin/env python3
"""Run deterministic, offline data and tiny-training smoke checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from _common import print_json

from hrm_particle.data import (  # noqa: E402
    generate_synthetic_dataset,
    validate_dataset_directory,
    verify_exact_answer,
)


def _data_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hrm-particle-smoke-") as temporary:
        output = Path(temporary) / "data"
        generate_synthetic_dataset(
            output,
            sizes={"train": 24, "dev": 12, "test": 12, "ood": 12},
            train_seed="offline-smoke-train",
            eval_seed="offline-smoke-eval-not-for-results",
            min_difficulty=1,
            max_difficulty=3,
        )
        result = validate_dataset_directory(output)
        if not verify_exact_answer(r"reasoning... \boxed{\frac{6}{8}}", "3/4"):
            raise AssertionError("exact verifier smoke check failed")
        return result["isolation"]


def _tiny_training_smoke() -> dict[str, Any]:
    """Exercise adapter PPO and detached Q supervision without model weights/network."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("PyTorch is required for the tiny-training smoke test") from exc

    from hrm_particle.adapter import ParticleAdapter, ParticleAdapterConfig, SharedQHead
    from hrm_particle.objectives import (
        anchor_rescue_advantages,
        clipped_token_policy_loss,
        supervised_q_loss,
    )

    torch.manual_seed(20260709)
    hidden_size = 16
    particle_count = 4
    time = 3
    adapter = ParticleAdapter(
        ParticleAdapterConfig(
            hidden_size=hidden_size,
            latent_size=8,
            bottleneck_size=8,
            max_relative_rms=0.10,
            initial_relative_rms=0.03,
        )
    )
    hidden = torch.randn(particle_count, time, hidden_size)
    query = torch.randn(particle_count, hidden_size)
    latents = torch.randn(particle_count, 8)
    latents[0].zero_()
    mask = torch.ones(particle_count, time, dtype=torch.bool)
    output = adapter(hidden, query, latents, mask)
    if not torch.equal(output.hidden_states[0], hidden[0]):
        raise AssertionError("z=0 anchor is not bit-exact in dummy training")
    if not bool((output.delta[1:].abs().sum() > 0).item()):
        raise AssertionError("explorer deltas are unexpectedly all zero")

    rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    advantages = anchor_rescue_advantages(rewards, alpha=0.2)
    # A differentiable toy log-probability derived from the adapter delta. The
    # labels remain external rewards; Q is deliberately absent from this loss.
    new_logprobs = output.delta.mean(dim=-1).reshape(1, particle_count, time)
    old_logprobs = new_logprobs.detach().clone()
    action_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
    action_mask[:, 0].zero_()
    actor_loss = clipped_token_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        action_mask,
        clip_epsilon=0.2,
    )
    actor_optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in adapter.parameters()]
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.loss.backward()
    actor_optimizer.step()
    if not any(not torch.equal(a, b) for a, b in zip(before, adapter.parameters())):
        raise AssertionError("dummy actor optimizer step changed no adapter parameter")

    q_head = SharedQHead(hidden_size=hidden_size, bottleneck_size=8)
    q_optimizer = torch.optim.AdamW(q_head.parameters(), lr=1e-3)
    terminal = output.hidden_states[:, -1].detach()
    q_logits = q_head(terminal, query.detach()).reshape(1, particle_count)
    q_loss = supervised_q_loss(q_logits, rewards, ranking_weight=0.1)
    q_optimizer.zero_grad(set_to_none=True)
    q_loss.loss.backward()
    q_optimizer.step()
    if any(parameter.grad is not None for parameter in adapter.parameters()):
        # Actor grads from the prior step are allowed to exist but Q must not
        # alter them. Clear and rerun the detached check explicitly.
        actor_optimizer.zero_grad(set_to_none=True)
        q_optimizer.zero_grad(set_to_none=True)
        detached_logits = q_head(terminal.detach(), query.detach()).reshape(1, particle_count)
        supervised_q_loss(detached_logits, rewards).loss.backward()
        if any(parameter.grad is not None for parameter in adapter.parameters()):
            raise AssertionError("Q supervision leaked gradients into the actor adapter")

    return {
        "torch_version": torch.__version__,
        "anchor_bit_exact": True,
        "explorer_relative_rms_max": float(output.relative_rms[1:].max().detach()),
        "actor_loss": float(actor_loss.loss.detach()),
        "q_loss": float(q_loss.loss.detach()),
        "q_ranking_pairs": q_loss.ranking_pairs,
    }


def _project_dummy_smoke() -> dict[str, Any] | None:
    """Run the optional full dummy rollout/update hook when supplied by core."""

    if importlib.util.find_spec("hrm_particle.smoke") is None:
        return None
    from hrm_particle.smoke import run_dummy_smoke
    result = run_dummy_smoke()
    if result is None:
        return {"status": "passed"}
    if isinstance(result, dict):
        return result
    return {"result": str(result)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-full-hook",
        action="store_true",
        help="fail if hrm_particle.smoke.run_dummy_smoke is unavailable",
    )
    parser.add_argument("--json-output", help="optional path for machine-readable results")
    args = parser.parse_args()
    try:
        result: dict[str, Any] = {
            "data": _data_smoke(),
            "tiny_training": _tiny_training_smoke(),
        }
        full = _project_dummy_smoke()
        if full is None:
            if args.require_full_hook:
                raise RuntimeError("hrm_particle.smoke.run_dummy_smoke is not installed")
            result["full_dummy_rollout"] = {"status": "not_available"}
        else:
            result["full_dummy_rollout"] = full
        result["status"] = "passed"
        if args.json_output:
            destination = Path(args.json_output).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print_json(result)
        return 0
    except (AssertionError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"smoke test failed: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
