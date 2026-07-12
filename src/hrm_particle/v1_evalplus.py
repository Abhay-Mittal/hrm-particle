"""Strict, dependency-light summarization of pinned EvalPlus result files."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PASS = "pass"
EXPECTED_FILES: Mapping[str, int] = {
    "ordinary-k4": 4,
    "particle-k4": 4,
    "particle-anchor": 1,
    "matched-zero-latent-k4": 4,
    "particle-q-selected": 1,
}


def _task_id_hash(task_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(task_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _estimate_pass_at_k(*, samples: int, correct: int, k: int) -> float:
    """HumanEval's unbiased pass@k estimator for one task."""

    if not 0 <= correct <= samples:
        raise ValueError("correct must lie in [0, samples]")
    if not 1 <= k <= samples:
        raise ValueError("k must lie in [1, samples]")
    if samples - correct < k:
        return 1.0
    product = 1.0
    for value in range(samples - correct + 1, samples + 1):
        product *= 1.0 - k / value
    return 1.0 - product


def _read_samples(path: Path, *, expected_samples: int) -> dict[str, list[str]]:
    by_task: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed sample {path}:{line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("solution"), str):
                raise ValueError(f"invalid sample schema {path}:{line_number}")
            by_task.setdefault(str(row.get("task_id")), []).append(row["solution"])
    if not by_task or any(len(values) != expected_samples for values in by_task.values()):
        raise ValueError(f"{path.name} does not contain exactly {expected_samples} samples per task")
    return by_task


def _read_result(
    path: Path,
    *,
    expected_samples: int,
    expected_solutions: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"EvalPlus result is missing: {path}. Run every pinned Docker command first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("eval"), dict):
        raise ValueError(f"invalid EvalPlus result schema: {path}")
    evaluation = payload["eval"]
    if not evaluation:
        raise ValueError(f"empty EvalPlus result: {path}")

    task_rows: dict[str, dict[str, Any]] = {}
    for raw_task_id, raw_results in evaluation.items():
        task_id = str(raw_task_id)
        if not isinstance(raw_results, list) or len(raw_results) != expected_samples:
            observed = len(raw_results) if isinstance(raw_results, list) else "non-list"
            raise ValueError(
                f"{path.name}:{task_id} has {observed} samples; expected {expected_samples}"
            )
        base_correct = 0
        plus_correct = 0
        observed_solutions: list[str] = []
        for result in raw_results:
            if not isinstance(result, dict):
                raise ValueError(f"{path.name}:{task_id} contains a non-object result")
            base_status = result.get("base_status")
            plus_status = result.get("plus_status")
            if not isinstance(result.get("solution"), str):
                raise ValueError(f"{path.name}:{task_id} has no solution text")
            observed_solutions.append(result["solution"])
            if base_status not in {PASS, "fail", "timeout", None}:
                raise ValueError(f"unknown EvalPlus base status {base_status!r}")
            if plus_status not in {PASS, "fail", "timeout", None}:
                raise ValueError(f"unknown EvalPlus plus status {plus_status!r}")
            base_pass = base_status == PASS
            # EvalPlus defines '+' success as passing both original and extra tests.
            plus_pass = base_pass and plus_status == PASS
            base_correct += int(base_pass)
            plus_correct += int(plus_pass)
        if task_id not in expected_solutions or observed_solutions != list(
            expected_solutions[task_id]
        ):
            raise ValueError(
                f"{path.name}:{task_id} does not correspond to the current generated samples"
            )
        task_rows[task_id] = {
            "samples": expected_samples,
            "base_correct": base_correct,
            "plus_correct": plus_correct,
        }

    metrics: dict[str, float | int | str] = {
        "tasks": len(task_rows),
        "samples_per_task": expected_samples,
        "dataset_hash": str(payload.get("hash", "")),
        "task_id_sha256": _task_id_hash(list(task_rows)),
    }
    for suite, field in (("base", "base_correct"), ("plus", "plus_correct")):
        for k in (1, expected_samples):
            estimates = [
                _estimate_pass_at_k(
                    samples=expected_samples,
                    correct=int(row[field]),
                    k=k,
                )
                for row in task_rows.values()
            ]
            value = math.fsum(estimates) / len(estimates)
            metrics[f"{suite}_pass_at_{k}"] = value
    return {"metrics": metrics, "tasks": task_rows}


def summarize_evalplus_directory(directory: Path) -> dict[str, Any]:
    """Validate and summarize all four MBPP+ arms from one generation run."""

    directory = Path(directory).expanduser().resolve()
    generation_path = directory / "generation-summary.json"
    if not generation_path.is_file():
        raise FileNotFoundError(f"generation summary is missing: {generation_path}")
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    expected_tasks = int(generation.get("tasks", 0))
    expected_ids = generation.get("task_ids")
    if expected_tasks <= 0:
        raise ValueError("generation summary has no positive task count")
    if expected_ids is not None:
        if not isinstance(expected_ids, list) or len(expected_ids) != expected_tasks:
            raise ValueError("generation summary task_ids do not match its task count")
        expected_id_set = {str(task_id) for task_id in expected_ids}
        if len(expected_id_set) != expected_tasks:
            raise ValueError("generation summary contains duplicate task IDs")
    else:
        expected_id_set = None

    arms: dict[str, Any] = {}
    common_ids: set[str] | None = None
    dataset_hash: str | None = None
    for name, samples_per_task in EXPECTED_FILES.items():
        sample_path = directory / f"{name}.jsonl"
        expected_hashes = generation.get("sample_sha256")
        if not isinstance(expected_hashes, Mapping) or expected_hashes.get(sample_path.name) != hashlib.sha256(
            sample_path.read_bytes()
        ).hexdigest():
            raise ValueError(f"generated sample checksum changed: {sample_path.name}")
        expected_solutions = _read_samples(
            sample_path, expected_samples=samples_per_task
        )
        result = _read_result(
            directory / f"{name}_eval_results.json",
            expected_samples=samples_per_task,
            expected_solutions=expected_solutions,
        )
        task_ids = set(result["tasks"])
        if len(task_ids) != expected_tasks:
            raise ValueError(
                f"{name} evaluated {len(task_ids)} tasks; expected {expected_tasks}"
            )
        if expected_id_set is not None and task_ids != expected_id_set:
            raise ValueError(f"{name} task IDs differ from generation-summary.json")
        if common_ids is None:
            common_ids = task_ids
        elif task_ids != common_ids:
            raise ValueError("EvalPlus arms contain different task sets")
        arm_hash = str(result["metrics"]["dataset_hash"])
        if not arm_hash:
            raise ValueError(f"{name} has no EvalPlus dataset hash")
        if dataset_hash is None:
            dataset_hash = arm_hash
        elif arm_hash != dataset_hash:
            raise ValueError("EvalPlus arms used different dataset hashes")
        arms[name] = result["metrics"]

    assert common_ids is not None
    expected_dataset_hash = generation.get("mbpp_dataset_hash")
    if expected_dataset_hash is not None and str(expected_dataset_hash) != dataset_hash:
        raise ValueError(
            "EvalPlus Docker dataset hash differs from the generation-time MBPP+ dataset"
        )
    summary = {
        "format": "hrm-particle-v1-evalplus-summary",
        "evalplus_version": generation.get("evalplus_version"),
        "evalplus_image": generation.get("evalplus_image"),
        "dataset": "MBPP+",
        "dataset_hash": dataset_hash,
        "tasks": expected_tasks,
        "task_id_sha256": _task_id_hash(list(common_ids)),
        "arms": arms,
        "comparisons": {
            "particle_minus_ordinary_plus_pass_at_4": (
                float(arms["particle-k4"]["plus_pass_at_4"])
                - float(arms["ordinary-k4"]["plus_pass_at_4"])
            ),
            "particle_minus_matched_zero_latent_plus_pass_at_4": (
                float(arms["particle-k4"]["plus_pass_at_4"])
                - float(arms["matched-zero-latent-k4"]["plus_pass_at_4"])
            ),
            "q_selected_minus_anchor_plus_pass_at_1": (
                float(arms["particle-q-selected"]["plus_pass_at_1"])
                - float(arms["particle-anchor"]["plus_pass_at_1"])
            ),
        },
    }
    destination = directory / "summary.json"
    temporary = destination.with_suffix(f".json.tmp-{__import__('os').getpid()}")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return summary


__all__ = [
    "EXPECTED_FILES",
    "summarize_evalplus_directory",
]
