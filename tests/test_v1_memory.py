from __future__ import annotations

import pytest

from hrm_particle.v1_memory import GIB, estimate_batch_from_peak, valid_batch_candidates


def test_candidates_preserve_target_prompts_per_rank() -> None:
    assert valid_batch_candidates(
        [8, 1, 4, 2, 2], target_prompts_per_rank_update=8
    ) == (1, 2, 4, 8)
    with pytest.raises(ValueError, match="include 1"):
        valid_batch_candidates([2, 4], target_prompts_per_rank_update=8)
    with pytest.raises(ValueError, match="divide"):
        valid_batch_candidates([1, 3], target_prompts_per_rank_update=8)


def test_estimator_uses_measured_bytes_not_gpu_names() -> None:
    plan = estimate_batch_from_peak(
        total_bytes=80 * GIB,
        baseline_bytes=8 * GIB,
        batch_one_peak_bytes=20 * GIB,
        candidates=[1, 2, 4, 8],
        target_prompts_per_rank_update=8,
        target_fraction=0.75,
        scaling_safety_factor=1.15,
    )
    assert plan.prompt_batch_size == 2
    assert plan.gradient_accumulation_steps == 4
    assert 0.0 < plan.estimated_fraction <= 0.75


def test_larger_measured_capacity_selects_larger_batch() -> None:
    plan = estimate_batch_from_peak(
        total_bytes=180 * GIB,
        baseline_bytes=10 * GIB,
        batch_one_peak_bytes=24 * GIB,
        candidates=[1, 2, 4, 8],
        target_prompts_per_rank_update=8,
        target_fraction=0.75,
    )
    assert plan.prompt_batch_size == 4
    assert plan.gradient_accumulation_steps == 2


@pytest.mark.parametrize("fraction", [0.69, 0.81])
def test_target_is_restricted_to_the_requested_70_to_80_percent(fraction: float) -> None:
    with pytest.raises(ValueError, match="target_fraction"):
        estimate_batch_from_peak(
            total_bytes=80 * GIB,
            baseline_bytes=8 * GIB,
            batch_one_peak_bytes=20 * GIB,
            candidates=[1, 2, 4, 8],
            target_prompts_per_rank_update=8,
            target_fraction=fraction,
        )
