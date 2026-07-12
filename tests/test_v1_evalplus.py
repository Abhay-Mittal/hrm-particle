from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_particle.v1_evalplus import EXPECTED_FILES, summarize_evalplus_directory


TASKS = ["Mbpp/1", "Mbpp/2"]


def _write_fixture(
    directory: Path,
    name: str,
    samples: int,
    outcomes: dict[str, tuple[int, int]],
) -> None:
    evaluation = {}
    samples_rows = []
    for task_id in TASKS:
        base_correct, plus_correct = outcomes[task_id]
        rows = []
        for index in range(samples):
            base_pass = index < base_correct
            plus_pass = index < plus_correct
            solution = f"{name}-{task_id}-{index}"
            samples_rows.append({"task_id": task_id, "solution": solution})
            rows.append(
                {
                    "task_id": task_id,
                    "solution": solution,
                    "base_status": "pass" if base_pass else "fail",
                    "plus_status": "pass" if plus_pass else "fail",
                }
            )
        evaluation[task_id] = rows
    sample_path = directory / f"{name}.jsonl"
    sample_path.write_text(
        "".join(json.dumps(row) + "\n" for row in samples_rows), encoding="utf-8"
    )
    (directory / f"{name}_eval_results.json").write_text(
        json.dumps({"hash": "dataset-md5", "eval": evaluation}), encoding="utf-8"
    )


def _complete_fixture(tmp_path: Path) -> None:
    for name, samples in EXPECTED_FILES.items():
        if name == "ordinary-k4":
            outcomes = {"Mbpp/1": (1, 1), "Mbpp/2": (0, 0)}
        elif name == "particle-k4":
            outcomes = {"Mbpp/1": (2, 1), "Mbpp/2": (1, 1)}
        elif name == "particle-anchor":
            outcomes = {"Mbpp/1": (1, 1), "Mbpp/2": (0, 0)}
        elif name == "matched-zero-latent-k4":
            outcomes = {"Mbpp/1": (1, 1), "Mbpp/2": (0, 0)}
        else:
            outcomes = {"Mbpp/1": (1, 1), "Mbpp/2": (1, 1)}
        _write_fixture(tmp_path, name, samples, outcomes)
    import hashlib

    sample_sha256 = {
        f"{name}.jsonl": hashlib.sha256((tmp_path / f"{name}.jsonl").read_bytes()).hexdigest()
        for name in EXPECTED_FILES
    }
    (tmp_path / "generation-summary.json").write_text(
        json.dumps(
            {
                "tasks": 2,
                "task_ids": TASKS,
                "evalplus_version": "0.3.1",
                "evalplus_image": "ganler/evalplus@sha256:fixture",
                "mbpp_dataset_hash": "dataset-md5",
                "sample_sha256": sample_sha256,
            }
        ),
        encoding="utf-8",
    )


def test_summarizer_computes_pass_at_1_and_pass_at_4(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    result = summarize_evalplus_directory(tmp_path)
    assert result["arms"]["ordinary-k4"]["plus_pass_at_1"] == pytest.approx(0.125)
    assert result["arms"]["ordinary-k4"]["plus_pass_at_4"] == pytest.approx(0.5)
    assert result["arms"]["particle-k4"]["plus_pass_at_4"] == pytest.approx(1.0)
    assert result["comparisons"]["particle_minus_ordinary_plus_pass_at_4"] == 0.5
    assert result["comparisons"]["particle_minus_matched_zero_latent_plus_pass_at_4"] == 0.5
    assert result["comparisons"]["q_selected_minus_anchor_plus_pass_at_1"] == 0.5
    assert (tmp_path / "summary.json").is_file()


def test_summarizer_rejects_missing_or_incomplete_results(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    (tmp_path / "particle-k4_eval_results.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        summarize_evalplus_directory(tmp_path)

    _complete_fixture(tmp_path)
    payload = json.loads((tmp_path / "ordinary-k4_eval_results.json").read_text())
    payload["eval"]["Mbpp/1"].pop()
    (tmp_path / "ordinary-k4_eval_results.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="expected 4"):
        summarize_evalplus_directory(tmp_path)


def test_summarizer_rejects_cross_arm_dataset_drift(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "particle-anchor_eval_results.json"
    payload = json.loads(path.read_text())
    payload["hash"] = "different-dataset"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different dataset hashes"):
        summarize_evalplus_directory(tmp_path)
