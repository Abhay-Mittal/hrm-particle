"""GPU-model-agnostic memory planning for the V1 runner.

The planner deliberately reasons from measured bytes, never from accelerator
names.  A real batch-one rollout/replay/backward supplies the fixed and
per-prompt memory estimate; the runner then validates the proposed batch on
every rank and uses the smallest safe result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


GIB = 2**30


@dataclass(frozen=True)
class EstimatedBatch:
    """Result of the conservative linear capacity estimate."""

    prompt_batch_size: int
    gradient_accumulation_steps: int
    estimated_peak_bytes: int
    estimated_fraction: float


def valid_batch_candidates(
    candidates: Iterable[int], *, target_prompts_per_rank_update: int
) -> tuple[int, ...]:
    """Return unique ascending candidates that preserve the global batch exactly."""

    target = int(target_prompts_per_rank_update)
    if target <= 0:
        raise ValueError("target_prompts_per_rank_update must be positive")
    normalized = sorted({int(value) for value in candidates})
    if not normalized or normalized[0] <= 0:
        raise ValueError("candidate prompt batch sizes must be positive")
    if any(target % value != 0 for value in normalized):
        raise ValueError(
            "every candidate prompt batch size must divide "
            "target_prompts_per_rank_update"
        )
    valid = tuple(normalized)
    if valid[0] != 1:
        raise ValueError(
            "candidate prompt batch sizes must include 1 and divide "
            "target_prompts_per_rank_update"
        )
    return valid


def estimate_batch_from_peak(
    *,
    total_bytes: int,
    baseline_bytes: int,
    batch_one_peak_bytes: int,
    candidates: Iterable[int],
    target_prompts_per_rank_update: int,
    target_fraction: float = 0.75,
    scaling_safety_factor: float = 1.15,
) -> EstimatedBatch:
    """Choose the largest candidate whose conservative estimate targets VRAM.

    ``baseline_bytes`` is memory held after loading the model and small
    trainables.  The measured batch-one increment is scaled linearly and then
    inflated by ``scaling_safety_factor``.  The selected batch is still only a
    proposal: the distributed runner validates it with the real workload.
    """

    total = int(total_bytes)
    baseline = int(baseline_bytes)
    peak_one = int(batch_one_peak_bytes)
    if total <= 0 or baseline < 0 or peak_one <= baseline or peak_one > total:
        raise ValueError("memory byte measurements are inconsistent")
    if not 0.70 <= float(target_fraction) <= 0.80:
        raise ValueError("target_fraction must lie in [0.70, 0.80]")
    if not math.isfinite(float(scaling_safety_factor)) or scaling_safety_factor < 1.0:
        raise ValueError("scaling_safety_factor must be finite and at least 1")

    valid = valid_batch_candidates(
        candidates,
        target_prompts_per_rank_update=target_prompts_per_rank_update,
    )
    incremental = peak_one - baseline
    budget = int(total * float(target_fraction))
    fits: list[tuple[int, int]] = []
    for batch in valid:
        estimated = baseline + math.ceil(incremental * batch * scaling_safety_factor)
        if batch == 1:
            # The empirical batch-one peak is more accurate than an inflated
            # extrapolation, and must remain available as the safe fallback.
            estimated = peak_one
        if estimated <= budget:
            fits.append((batch, estimated))
    if not fits:
        raise RuntimeError(
            "even prompt batch size 1 exceeds the configured VRAM target; "
            "shorten sequences or use a GPU with more usable memory"
        )
    batch, estimated = fits[-1]
    target = int(target_prompts_per_rank_update)
    return EstimatedBatch(
        prompt_batch_size=batch,
        gradient_accumulation_steps=target // batch,
        estimated_peak_bytes=estimated,
        estimated_fraction=estimated / total,
    )


__all__ = [
    "EstimatedBatch",
    "GIB",
    "estimate_batch_from_peak",
    "valid_batch_candidates",
]
