"""Evaluation metrics that separate candidate generation from Q selection."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .rollout import ParticleRollout, ParticleRolloutEngine, RolloutExample


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Dict[str, float]
    per_prompt: List[Dict[str, Any]]


@dataclass(frozen=True)
class BootstrapInterval:
    mean_delta: float
    low: float
    high: float
    confidence: float


def _score_q(q_head: nn.Module, rollout: ParticleRollout) -> Tensor:
    b, k, hidden = rollout.terminal_states.shape
    terminal = rollout.terminal_states.detach().reshape(b * k, hidden)
    prompt = rollout.prompt_summary.detach().reshape(b * k, -1)
    try:
        parameter = next(q_head.parameters())
    except StopIteration:
        parameter = None
    if parameter is not None:
        terminal = terminal.to(device=parameter.device, dtype=parameter.dtype)
        prompt = prompt.to(device=parameter.device, dtype=parameter.dtype)
    try:
        logits = q_head(terminal, prompt)
    except TypeError:
        logits = q_head(terminal_state=terminal, prompt_summary=prompt)
    if isinstance(logits, dict):
        logits = logits.get("logits", logits.get("q_logits"))
    elif hasattr(logits, "logits"):
        logits = logits.logits
    if logits is None:
        raise RuntimeError("Q head did not return logits")
    return logits.reshape(b, k)


def evaluate_rollout(
    rollout: ParticleRollout,
    q_logits: Optional[Tensor] = None,
) -> EvaluationResult:
    """Compute generation, rescue, and optional selector metrics.

    Oracle pass@K assesses the particle generator independently of Q.  Q-selected
    accuracy and capture assess the selector separately; neither is substituted
    for the other.
    """

    rewards = rollout.rewards.float()
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rollout rewards must be [batch, K>=2]")
    b, k = rewards.shape
    if q_logits is not None and q_logits.shape != rewards.shape:
        raise ValueError("q_logits must match reward shape")

    anchor = rewards[:, 0]
    explorers = rewards[:, 1:]
    oracle = rewards.max(dim=1).values
    mean_candidate = rewards.mean(dim=1)
    anchor_wrong = anchor < 0.5
    rescued = (explorers.max(dim=1).values > 0.5) & anchor_wrong
    mixed = (rewards.sum(dim=1) > 0) & (rewards.sum(dim=1) < k)
    generated_lengths = rollout.generated_mask.float().sum(dim=-1)

    duplicate_fractions = []
    all_identical = []
    for group in rollout.response_texts:
        normalized = [" ".join(text.lower().split()) for text in group]
        unique = len(set(normalized))
        duplicate_fractions.append((k - unique) / float(k - 1))
        all_identical.append(float(unique == 1))

    verification_flat = [item for group in rollout.verification for item in group]
    parseable = [
        float(item.predicted is not None and item.expected is not None)
        for item in verification_flat
    ]
    metrics: Dict[str, float] = {
        "num_prompts": float(b),
        "k": float(k),
        "anchor_accuracy": float(anchor.mean()),
        "explorer_accuracy": float(explorers.mean()),
        "mean_candidate_accuracy": float(mean_candidate.mean()),
        "oracle_pass_at_k": float(oracle.mean()),
        "anchor_wrong_count": float(anchor_wrong.sum()),
        "rescue_count": float(rescued.sum()),
        "rescue_given_anchor_wrong": float(
            rescued[anchor_wrong].float().mean() if bool(anchor_wrong.any()) else 0.0
        ),
        "mixed_group_fraction": float(mixed.float().mean()),
        "mean_generated_tokens": float(generated_lengths.mean()),
        "duplicate_candidate_fraction": float(
            sum(duplicate_fractions) / len(duplicate_fractions)
        ),
        "all_candidates_identical_fraction": float(sum(all_identical) / len(all_identical)),
        "parseable_fraction": float(sum(parseable) / len(parseable)),
    }

    selected_indices: Optional[Tensor] = None
    if q_logits is not None:
        selected_indices = q_logits.argmax(dim=1)
        selected = rewards.gather(1, selected_indices[:, None]).squeeze(1)
        selected_accuracy = selected.mean()
        denominator = oracle.mean() - mean_candidate.mean()
        capture = (
            (selected_accuracy - mean_candidate.mean()) / denominator
            if float(denominator.abs()) > 1e-8
            else denominator.new_tensor(0.0)
        )
        pair_correct = 0.0
        pair_count = 0
        for group_logits, group_rewards in zip(q_logits, rewards):
            positive = group_logits[group_rewards > 0.5]
            negative = group_logits[group_rewards <= 0.5]
            if positive.numel() and negative.numel():
                comparisons = positive[:, None] > negative[None, :]
                pair_correct += float(comparisons.float().sum())
                pair_count += comparisons.numel()
        probabilities = q_logits.sigmoid()
        brier = (probabilities - rewards).square().mean()
        ece = probabilities.new_tensor(0.0)
        flat_probabilities = probabilities.flatten()
        flat_rewards = rewards.flatten()
        for lower in torch.linspace(0.0, 0.9, 10, device=probabilities.device):
            upper = lower + 0.1
            in_bin = (flat_probabilities >= lower) & (
                flat_probabilities <= upper if float(upper) >= 1.0 else flat_probabilities < upper
            )
            if bool(in_bin.any()):
                weight = in_bin.float().mean()
                ece = ece + weight * (
                    flat_probabilities[in_bin].mean() - flat_rewards[in_bin].mean()
                ).abs()
        oracle_success = oracle > 0.5
        flat_positive = q_logits.flatten()[rewards.flatten() > 0.5]
        flat_negative = q_logits.flatten()[rewards.flatten() <= 0.5]
        if flat_positive.numel() and flat_negative.numel():
            auc_comparisons = flat_positive[:, None] - flat_negative[None, :]
            auroc = (auc_comparisons.gt(0).float() + 0.5 * auc_comparisons.eq(0).float()).mean()
        else:
            auroc = q_logits.new_tensor(0.0)
        metrics.update(
            {
                "q_selected_accuracy": float(selected_accuracy),
                "q_capture_at_k": float(capture),
                "q_within_prompt_pair_accuracy": (
                    pair_correct / pair_count if pair_count else 0.0
                ),
                "q_pair_count": float(pair_count),
                "q_oracle_regret": float(oracle.mean() - selected_accuracy),
                "q_brier": float(brier),
                "q_ece_10bin": float(ece),
                "q_auroc": float(auroc),
                "q_selected_given_oracle_success": float(
                    selected[oracle_success].mean() if bool(oracle_success.any()) else 0.0
                ),
                "q_selected_mixed_accuracy": float(
                    selected[mixed].mean() if bool(mixed.any()) else 0.0
                ),
            }
        )

    per_prompt: List[Dict[str, Any]] = []
    for index in range(b):
        record: Dict[str, Any] = {
            "example_id": rollout.example_ids[index],
            "reference": rollout.references[index],
            "anchor_correct": float(anchor[index]),
            "mean_candidate_accuracy": float(mean_candidate[index]),
            "oracle_correct": float(oracle[index]),
            "rescued": float(rescued[index]),
            "response_texts": list(rollout.response_texts[index]),
            "rewards": [float(value) for value in rewards[index].tolist()],
            "generated_tokens": [
                int(value) for value in generated_lengths[index].tolist()
            ],
            "verification": [
                {
                    "correct": bool(item.correct),
                    "predicted": str(item.predicted) if item.predicted is not None else None,
                    "expected": str(item.expected) if item.expected is not None else None,
                    "predicted_text": item.predicted_text,
                    "expected_text": item.expected_text,
                    "error": item.error,
                }
                for item in rollout.verification[index]
            ],
        }
        if selected_indices is not None:
            branch = int(selected_indices[index])
            record["q_selected_branch"] = float(branch)
            record["q_selected_correct"] = float(rewards[index, branch])
            record["q_logits"] = [float(value) for value in q_logits[index].tolist()]
        per_prompt.append(record)
    return EvaluationResult(metrics=metrics, per_prompt=per_prompt)


@torch.no_grad()
def evaluate_examples(
    engine: ParticleRolloutEngine,
    examples: Sequence[RolloutExample],
    *,
    q_head: Optional[nn.Module] = None,
    batch_size: int = 8,
    deadline_unix: Optional[float] = None,
) -> EvaluationResult:
    """Evaluate examples in bounded rollout batches and aggregate per-prompt data."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records: List[Dict[str, Any]] = []
    rollouts: List[tuple[ParticleRollout, Optional[Tensor]]] = []
    for start in range(0, len(examples), batch_size):
        if deadline_unix is not None and time.time() >= deadline_unix:
            raise RuntimeError("evaluation deadline reached before completing the split")
        rollout = engine.generate(examples[start : start + batch_size])
        logits = _score_q(q_head, rollout) if q_head is not None else None
        rollouts.append((rollout, logits))
        records.extend(evaluate_rollout(rollout, logits).per_prompt)
    if not rollouts:
        raise ValueError("examples must be non-empty")

    # Metrics are recomputed from prompt-level/candidate-level sufficient data
    # rather than averaging batch means with unequal final batch sizes.
    reward_groups = torch.cat([rollout.rewards.cpu() for rollout, _ in rollouts], dim=0)
    q_groups = (
        torch.cat([logits.detach().cpu() for _, logits in rollouts if logits is not None], dim=0)
        if q_head is not None
        else None
    )
    # Variable response lengths prevent concatenating full rollout tensors, so
    # aggregate scalar metrics using prompt-weighted batch results.  Candidate
    # and parseability denominators are constant K and therefore equivalent.
    weighted: Dict[str, float] = {}
    total = 0
    for rollout, logits in rollouts:
        result = evaluate_rollout(rollout, logits)
        weight = rollout.batch_size
        total += weight
        for name, value in result.metrics.items():
            if name in {"num_prompts", "k", "anchor_wrong_count", "rescue_count", "q_pair_count"}:
                continue
            weighted[name] = weighted.get(name, 0.0) + value * weight
    metrics = {name: value / total for name, value in weighted.items()}
    metrics["num_prompts"] = float(total)
    metrics["k"] = float(reward_groups.shape[1])
    metrics["anchor_wrong_count"] = float((reward_groups[:, 0] < 0.5).sum())
    metrics["rescue_count"] = float(
        (((reward_groups[:, 1:].max(dim=1).values > 0.5) & (reward_groups[:, 0] < 0.5))).sum()
    )
    wrong = metrics["anchor_wrong_count"]
    metrics["rescue_given_anchor_wrong"] = metrics["rescue_count"] / wrong if wrong else 0.0
    if q_groups is not None:
        oracle = reward_groups.max(dim=1).values
        mean_candidate = reward_groups.mean(dim=1)
        selected = reward_groups.gather(1, q_groups.argmax(dim=1, keepdim=True)).squeeze(1)
        denominator = oracle.mean() - mean_candidate.mean()
        metrics["q_selected_accuracy"] = float(selected.mean())
        metrics["q_capture_at_k"] = float(
            (selected.mean() - mean_candidate.mean()) / denominator
            if float(denominator.abs()) > 1e-8
            else 0.0
        )
        metrics["q_oracle_regret"] = float(oracle.mean() - selected.mean())
        probabilities = q_groups.sigmoid()
        metrics["q_brier"] = float((probabilities - reward_groups).square().mean())
        flat_p = probabilities.flatten()
        flat_y = reward_groups.flatten()
        ece = 0.0
        for bin_index in range(10):
            lower = bin_index / 10.0
            upper = (bin_index + 1) / 10.0
            in_bin = (flat_p >= lower) & (flat_p <= upper if bin_index == 9 else flat_p < upper)
            if bool(in_bin.any()):
                ece += float(in_bin.float().mean()) * float(
                    (flat_p[in_bin].mean() - flat_y[in_bin].mean()).abs()
                )
        metrics["q_ece_10bin"] = ece
        pair_correct = 0.0
        pair_count = 0
        for group_logits, group_rewards in zip(q_groups, reward_groups):
            positive = group_logits[group_rewards > 0.5]
            negative = group_logits[group_rewards <= 0.5]
            if positive.numel() and negative.numel():
                comparisons = positive[:, None] > negative[None, :]
                pair_correct += float(comparisons.float().sum())
                pair_count += comparisons.numel()
        metrics["q_within_prompt_pair_accuracy"] = (
            pair_correct / pair_count if pair_count else 0.0
        )
        metrics["q_pair_count"] = float(pair_count)
        flat_positive = q_groups.flatten()[reward_groups.flatten() > 0.5]
        flat_negative = q_groups.flatten()[reward_groups.flatten() <= 0.5]
        if flat_positive.numel() and flat_negative.numel():
            differences = flat_positive[:, None] - flat_negative[None, :]
            metrics["q_auroc"] = float(
                (differences.gt(0).float() + 0.5 * differences.eq(0).float()).mean()
            )
        else:
            metrics["q_auroc"] = 0.0
        mixed = (reward_groups.sum(dim=1) > 0) & (reward_groups.sum(dim=1) < reward_groups.shape[1])
        oracle_success = oracle > 0.5
        metrics["q_selected_given_oracle_success"] = float(
            selected[oracle_success].mean() if bool(oracle_success.any()) else 0.0
        )
        metrics["q_selected_mixed_accuracy"] = float(
            selected[mixed].mean() if bool(mixed.any()) else 0.0
        )
    return EvaluationResult(metrics=metrics, per_prompt=records)


def paired_bootstrap_delta(
    system_scores: Sequence[float] | Tensor,
    baseline_scores: Sequence[float] | Tensor,
    *,
    num_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Prompt-paired bootstrap interval for an accuracy (or reward) delta."""

    system = torch.as_tensor(system_scores, dtype=torch.float64)
    baseline = torch.as_tensor(baseline_scores, dtype=torch.float64)
    if system.ndim != 1 or system.shape != baseline.shape or system.numel() == 0:
        raise ValueError("scores must be non-empty, same-length vectors")
    if num_samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    differences = system - baseline
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        differences.numel(),
        (num_samples, differences.numel()),
        generator=generator,
    )
    samples = differences[indices].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        mean_delta=float(differences.mean()),
        low=float(torch.quantile(samples, tail)),
        high=float(torch.quantile(samples, 1.0 - tail)),
        confidence=confidence,
    )


def evaluate_from_config(
    config: Mapping[str, Any],
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load a trained adapter and run particle plus matched clean-sampling eval."""

    from .checkpoint import load_checkpoint
    from .data import load_jsonl
    from .trainer import _project_path, _rollout_engine_from_config, build_components_from_config

    evaluation = dict(config.get("evaluation", {}))
    if checkpoint is None and not evaluation.get("allow_untrained_adapter", False):
        raise ValueError(
            "evaluation requires --checkpoint; set evaluation.allow_untrained_adapter=true "
            "only for an explicitly labeled step-0 adapter baseline"
        )
    model, tokenizer = build_components_from_config(config)
    particle_engine = _rollout_engine_from_config(
        model, tokenizer, config, require_k4=False
    )
    # Evaluation never needs reference-policy replay log-probabilities.
    particle_engine.config = replace(
        particle_engine.config, compute_reference_logprobs=False
    )
    loaded_step = 0
    if checkpoint is not None:
        loaded = load_checkpoint(
            checkpoint,
            adapter=model.adapter,
            q_head=model.q_head,
            map_location=next(model.adapter.parameters()).device,
            restore_rng=False,
        )
        loaded_step = int(loaded["step"])
        checkpoint_config = loaded.get("config")
        if checkpoint_config is None:
            raise ValueError("checkpoint lacks architecture provenance")

        def architecture_signature(value: Mapping[str, Any]) -> dict[str, Any]:
            model_section = dict(value.get("model", {}))
            adapter_section = dict(value.get("adapter", {}))
            q_section = dict(value.get("q_head", {}))
            prompting_section = dict(value.get("prompting", {}))
            generation_section = dict(value.get("generation", {}))
            return {
                "model": model_section.get("pretrained_model_name_or_path"),
                "revision": model_section.get("revision", "main"),
                "dtype": model_section.get("dtype", "bfloat16"),
                "latent_size": int(adapter_section.get("latent_size", 64)),
                "adapter_bottleneck": int(adapter_section.get("bottleneck_size", 64)),
                "max_relative_rms": float(adapter_section.get("max_relative_rms", 0.10)),
                "q_bottleneck": int(q_section.get("bottleneck_size", 256)),
                "condition": prompting_section.get("condition", "synth,cot"),
                "first_token_mode": generation_section.get(
                    "first_token_mode", "causal_prefix"
                ),
                "response_prefix": generation_section.get(
                    "response_prefix", "\nSolution:\n"
                ),
            }

        if architecture_signature(checkpoint_config) != architecture_signature(config):
            raise ValueError("evaluation config is incompatible with checkpoint architecture")
        checkpoint_commit = loaded.get("metadata", {}).get("provenance", {}).get(
            "resolved_revision"
        )
        current_commit = getattr(
            getattr(getattr(model, "base_model", None), "config", None),
            "_commit_hash",
            None,
        )
        if checkpoint_commit is not None and current_commit != checkpoint_commit:
            raise ValueError("resolved Hugging Face model revision differs from checkpoint")
    model.eval()

    # Matched ordinary baseline: same prompt, response prefix, decoding, token
    # budget and K, but every latent is exactly zero. Particle zero stays greedy;
    # the other three branches are ordinary stochastic samples.
    def zero_latents(batch_size, k, latent_dim, **kwargs):
        return torch.zeros(
            batch_size,
            k,
            latent_dim,
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    ordinary_engine = ParticleRolloutEngine(
        model,
        tokenizer,
        verifier=particle_engine.verifier,
        config=replace(
            particle_engine.config,
            compute_reference_logprobs=False,
        ),
        latent_sampler=zero_latents,
    )
    data_config = dict(config.get("data", {}))
    data_directory = _project_path(str(data_config.get("directory", "data/processed")))
    destination = _project_path(
        output_dir or config.get("output", {}).get("directory", "runs/eval")
    )
    destination.mkdir(parents=True, exist_ok=True)
    batch_size = int(evaluation.get("prompt_batch_size", 1))
    maximum = evaluation.get("max_examples_per_split")
    # A separate evaluation process normally does not inherit training's
    # deadline. Keep a conservative default guard unless explicitly tightened.
    configured_minutes = float(evaluation.get("max_wall_minutes", 120.0))
    if configured_minutes <= 0:
        raise ValueError("evaluation.max_wall_minutes must be positive")
    deadline_unix = time.time() + 60.0 * configured_minutes
    deadline_text = os.environ.get("HRM_PARTICLE_DEADLINE_UNIX")
    if deadline_text:
        deadline_unix = min(deadline_unix, float(deadline_text))
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint is not None else None

    def file_hash(path: Optional[Path]) -> Optional[str]:
        if path is None or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "not-installed"
    results: dict[str, Any] = {
        "checkpoint_step": loaded_step,
        "provenance": {
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_sha256": file_hash(checkpoint_path),
            "untrained_adapter_baseline": checkpoint is None,
            "config": dict(config),
            "manifest_sha256": file_hash(data_directory / "manifest.json"),
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "requested_revision": config.get("model", {}).get("revision", "main"),
            "max_wall_minutes": configured_minutes,
            "resolved_revision": getattr(
                getattr(getattr(model, "base_model", None), "config", None),
                "_commit_hash",
                None,
            ),
        },
        "splits": {},
    }
    for split in evaluation.get("splits", ["test", "ood"]):
        filename = data_config.get(f"{split}_file", f"{split}.jsonl")
        records = load_jsonl(data_directory / str(filename))
        if maximum is not None:
            records = records[: int(maximum)]
        examples = [
            RolloutExample(
                prompt=record.prompt,
                answer=record.answer,
                example_id=record.id,
                metadata=record.metadata,
            )
            for record in records
        ]
        particle = evaluate_examples(
            particle_engine,
            examples,
            q_head=model.q_head,
            batch_size=batch_size,
            deadline_unix=deadline_unix,
        )
        ordinary = evaluate_examples(
            ordinary_engine,
            examples,
            batch_size=batch_size,
            deadline_unix=deadline_unix,
        )
        particle_oracle = [float(row["oracle_correct"]) for row in particle.per_prompt]
        ordinary_oracle = [float(row["oracle_correct"]) for row in ordinary.per_prompt]
        interval = paired_bootstrap_delta(
            particle_oracle,
            ordinary_oracle,
            num_samples=int(evaluation.get("bootstrap_samples", 2_000)),
            confidence=float(evaluation.get("confidence_level", 0.95)),
            seed=int(config.get("seed", 0)),
        )
        split_result = {
            "particle": particle.metrics,
            "ordinary_zero_latent": ordinary.metrics,
            "paired_oracle_delta": {
                "mean": interval.mean_delta,
                "low": interval.low,
                "high": interval.high,
                "confidence": interval.confidence,
            },
        }
        results["splits"][str(split)] = split_result
        (destination / f"{split}-metrics.json").write_text(
            json.dumps(split_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if evaluation.get("save_candidate_text", True):
            with (destination / f"{split}-per-prompt.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for particle_row, ordinary_row in zip(
                    particle.per_prompt, ordinary.per_prompt
                ):
                    handle.write(
                        json.dumps(
                            {"particle": particle_row, "ordinary": ordinary_row},
                            sort_keys=True,
                        )
                        + "\n"
                    )
    (destination / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results


__all__ = [
    "BootstrapInterval",
    "EvaluationResult",
    "evaluate_examples",
    "evaluate_from_config",
    "evaluate_rollout",
    "paired_bootstrap_delta",
]
