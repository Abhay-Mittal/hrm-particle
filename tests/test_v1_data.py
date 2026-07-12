from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hrm_particle.v1_data import (
    PRESETS,
    SOURCE_SPECS,
    V1Preset,
    V1Record,
    build_ngram_index,
    normalize_dapo_row,
    normalize_hendrycks_math_row,
    normalize_rows,
    normalize_verified_python_row,
    prepare_v1_dataset,
    select_records,
    validate_python_verification_info,
    validate_v1_directory,
    _select_eval_records,
)


def test_gsm_symbolic_uses_the_supported_p1_configuration() -> None:
    assert SOURCE_SPECS["gsm_symbolic_eval"].configs == ("p1",)


def test_full_gsm_symbolic_subset_is_five_instances_per_template() -> None:
    from hrm_particle.v1_data import normalize_gsm_symbolic_row

    symbolic = [
        normalize_gsm_symbolic_row(
            {
                "question": _math_words(f"symbolic-{template}", instance),
                "answer": f"work\n#### {template + instance}",
                "original_id": template,
                "instance": instance,
            }
        )
        for template in range(100)
        for instance in range(6)
    ]
    selected, stats = _select_eval_records(
        {
            "gsm8k_eval": [],
            "math500_eval": [],
            "gsm_symbolic_eval": symbolic,
        },
        V1Preset(
            q_warm={},
            rl_train={},
            eval={"gsm8k_eval": 0, "math500_eval": 0, "gsm_symbolic_eval": 500},
        ),
    )
    assert len(selected) == 500
    by_template: dict[int, set[int]] = {}
    for record in selected:
        by_template.setdefault(int(record.metadata["original_id"]), set()).add(
            int(record.metadata["instance"])
        )
    assert len(by_template) == 100
    assert all(instances == {0, 1, 2, 3, 4} for instances in by_template.values())
    assert stats["gsm_symbolic_eval"]["original_templates"] == 100
    assert all(
        stats["gsm_symbolic_eval"][f"instance_{instance}"] == 100
        for instance in range(5)
    )


def _math_words(kind: str, index: int) -> str:
    return (
        f"{kind} item {index} asks a {kind} learner to combine several carefully "
        f"chosen {kind} quantities and explain the final numerical result token {kind}{index}."
    )


def _code_row(index: int, *, difficulty: str = "easy", reward: float = 1.0) -> dict:
    return {
        "source": "apps",
        "task_type": "verifiable_code",
        "in_source_id": str(index),
        "problem": (
            f"Write a Python program for fixture {index} that reads one integer, "
            f"adds the fixture constant {index}, and prints the resulting integer."
        ),
        "gold_standard_solution": f"x = int(input())\nprint(x + {index})",
        "problem_id": f"code-{index}",
        "metadata": {"difficulty": difficulty, "problem_url": f"https://example/{index}"},
        "verification_info": {
            "language": "python",
            "test_cases": [
                {
                    "fn_name": None,
                    "input": "2\n",
                    "output": f"{index + 2}\n",
                    "type": "stdin_stdout",
                }
            ],
        },
        "test_reward": reward,
    }


def _fixtures() -> dict[str, list[dict]]:
    return {
        "gsm8k_train": [
            {
                "id": f"gsm-train-{i}",
                "question": _math_words("gsmtrain", i),
                "answer": f"work\n#### {10 + i}",
            }
            for i in range(4)
        ],
        "hendrycks_math_train": [
            {
                "id": f"math-train-{i}",
                "problem": _math_words("mathtrain", i),
                "solution": rf"Reasoning gives \boxed{{{20 + i}}}.",
                "level": f"Level {(i % 3) + 1}",
                "type": "Algebra",
            }
            for i in range(4)
        ],
        "dapo_math_en": [
            {
                "prompt": _math_words("dapo", i),
                "solution": str(30 + i),
                "data_source": "math_dapo",
                "ability": "MATH",
                "reward_model": {
                    "ground_truth": str(30 + i),
                    "style": "rule-lighteval/MATH_v2",
                },
                "extra_info": {"index": f"dapo-{i}"},
            }
            for i in range(8)
        ],
        "verified_python": [_code_row(i) for i in range(6)],
        "gsm8k_eval": [
            {
                "id": f"gsm-eval-{i}",
                "question": _math_words("gsmeval", i),
                "answer": f"evaluation work\n#### {40 + i}",
            }
            for i in range(3)
        ],
        "math500_eval": [
            {
                "unique_id": f"math500-{i}",
                "problem": _math_words("mathfivehundred", i),
                "solution": rf"Evaluation reasoning \boxed{{{50 + i}}}.",
                "answer": str(50 + i),
                "subject": "Algebra",
                "level": (i % 5) + 1,
            }
            for i in range(3)
        ],
        "gsm_symbolic_eval": [
            {
                "question": _math_words("symbolic", i),
                "answer": f"symbolic work\n#### {60 + i}",
                "original_id": i,
                "instance": 0,
            }
            for i in range(3)
        ],
    }


def test_full_preset_has_requested_q_and_rl_counts():
    full = PRESETS["full"]
    assert full.q_warm == {
        "gsm8k_train": 512,
        "hendrycks_math_train": 512,
        "dapo_math_en": 512,
        "verified_python": 512,
    }
    assert full.q_prompt_groups == 2048
    assert full.q_prompt_groups * 4 == 8192
    assert full.rl_train == {"dapo_math_en": 4000, "verified_python": 1000}


def test_poc_preset_keeps_q_scale_but_trims_rl_and_evaluation() -> None:
    poc = PRESETS["poc"]
    assert poc.q_prompt_groups == 2048
    assert poc.q_prompt_groups * 4 == 8192
    assert poc.rl_train == {"dapo_math_en": 2400, "verified_python": 600}
    assert poc.eval == {
        "gsm8k_eval": 500,
        "math500_eval": 200,
        "gsm_symbolic_eval": 200,
    }


def test_v1_record_schema_round_trip_is_strict():
    record = normalize_dapo_row(_fixtures()["dapo_math_en"][0])
    assert V1Record.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError, match="unexpected V1Record fields"):
        V1Record.from_dict({**record.to_dict(), "typo": 1})
    with pytest.raises(ValueError, match="missing V1Record fields"):
        payload = record.to_dict()
        payload.pop("answer")
        V1Record.from_dict(payload)


def test_hendrycks_q_warm_rejects_harder_levels_and_unboxed_answers():
    easy = {
        "problem": "Find x from a simple fixture equation.",
        "solution": r"Thus \boxed{7}.",
        "level": "Level 3",
        "type": "Algebra",
    }
    hard = {**easy, "level": "Level 4"}
    unboxed = {**easy, "solution": "The prose says seven but has no reference box."}
    records, stats = normalize_rows(
        [easy, hard, unboxed], normalize_hendrycks_math_row, pool="q_warm"
    )
    assert [record.answer for record in records] == ["7"]
    assert stats["hard_difficulty"] == 1
    assert stats["missing_answer"] == 1


def test_verified_python_requires_easy_verified_python_tests_without_execution():
    accepted = normalize_verified_python_row(_code_row(1))
    assert accepted.task_type == "code"
    assert accepted.verification_info["language"] == "python"

    rows = [
        _code_row(2, difficulty="very_hard"),
        _code_row(3, reward=0.0),
        {**_code_row(4), "verification_info": {"language": "javascript", "test_cases": []}},
    ]
    records, stats = normalize_rows(rows, normalize_verified_python_row, pool="rl_train")
    assert records == []
    assert stats["hard_difficulty"] == 1
    assert stats["not_verified"] == 1
    assert stats["non_python"] == 1

    with pytest.raises(ValueError, match="non-empty list"):
        validate_python_verification_info({"language": "python", "test_cases": []})
    functional = _code_row(5)
    functional["verification_info"]["test_cases"][0].update(
        {"type": "functional", "fn_name": "solve"}
    )
    records, stats = normalize_rows(
        [functional], normalize_verified_python_row, pool="rl_train"
    )
    assert records == []
    assert stats["invalid_test_cases"] == 1


def test_13_token_decontamination_is_exact_and_deterministic():
    evaluation = "zero one two three four five six seven eight nine ten eleven twelve thirteen"
    index = build_ngram_index([evaluation], n=13)
    assert index.overlaps("prefix zero one two three four five six seven eight nine ten eleven twelve suffix")
    assert not index.overlaps("zero one two three four five six seven eight nine ten eleven changed")

    candidates = [normalize_dapo_row(row) for row in _fixtures()["dapo_math_en"][:2]]
    contaminated = V1Record.from_dict(
        {
            **candidates[0].to_dict(),
            "id": "contaminated-id",
            "prompt": evaluation,
        }
    )
    chosen, _ = select_records(
        [contaminated, *candidates],
        count=2,
        seed="selection",
        pool="rl_train",
        contamination_index=index,
    )
    assert all(record.id != "contaminated-id" for record in chosen)


def test_offline_prepare_writes_expected_atomic_files_and_checksums(tmp_path):
    output = tmp_path / "v1"
    result = prepare_v1_dataset(
        output,
        preset="smoke",
        seed="offline-seed",
        source_rows=_fixtures(),
        revisions={
            spec: f"fixture-sha-{index}"
            for index, spec in enumerate(
                sorted(
                    {
                        "openai/gsm8k",
                        "EleutherAI/hendrycks_math",
                        "open-r1/DAPO-Math-17k-Processed",
                        "open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled",
                        "HuggingFaceH4/MATH-500",
                        "apple/GSM-Symbolic",
                    }
                )
            )
        },
    )
    assert (output / "q_warm.jsonl").is_file()
    assert (output / "rl_train.jsonl").is_file()
    assert (output / "eval_math.jsonl").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "manifest.sha256").is_file()
    assert result["files"]["q_warm"]["records"] == 8
    assert result["files"]["rl_train"]["records"] == 6
    assert result["files"]["eval"]["records"] == 6
    assert result["q_warm"]["candidates_at_k4"] == 32
    assert result["safety"]["code_executed_during_preparation"] is False
    validated = validate_v1_directory(output)
    assert validated["records"] == 20

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert all(
        source["revision"].startswith("fixture-sha-")
        for source in manifest["sources"].values()
    )
    assert manifest["decontamination"]["ngram_size"] == 13


def test_offline_output_is_byte_deterministic(tmp_path):
    first = prepare_v1_dataset(
        tmp_path / "first", preset="smoke", seed="same", source_rows=_fixtures()
    )
    second = prepare_v1_dataset(
        tmp_path / "second", preset="smoke", seed="same", source_rows=_fixtures()
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["files"] == second["files"]


def test_resolver_and_loader_are_injectable_without_network(tmp_path):
    fixtures = _fixtures()
    resolved: list[str] = []
    loaded: list[tuple[str, str]] = []

    def resolver(repo_id: str) -> str:
        resolved.append(repo_id)
        return "a" * 40

    def loader(spec, revision):
        loaded.append((spec.key, revision))
        return fixtures[spec.key]

    prepare_v1_dataset(
        tmp_path / "injected",
        preset="smoke",
        seed="injected",
        revision_resolver=resolver,
        dataset_loader=loader,
    )
    assert len(resolved) == 6  # GSM8K train/test share one pinned repository revision.
    assert {key for key, _ in loaded} == set(fixtures)
    assert all(revision == "a" * 40 for _, revision in loaded)


def test_optional_mbpp_prompts_are_included_via_injected_lazy_loader(tmp_path):
    result = prepare_v1_dataset(
        tmp_path / "mbpp",
        preset="smoke",
        seed="mbpp",
        source_rows=_fixtures(),
        include_mbpp_plus_contamination=True,
        mbpp_prompt_loader=lambda: (["Implement a tiny deterministic MBPP fixture function."], "9.9"),
    )
    assert result["decontamination"]["includes_mbpp_plus"] is True
    assert result["decontamination"]["mbpp_plus_prompts"] == 1
    assert result["decontamination"]["evalplus_version"] == "9.9"


@pytest.mark.parametrize("filename", ["q_warm.jsonl", "rl_train.jsonl", "eval_math.jsonl"])
def test_validate_detects_every_jsonl_tampering(tmp_path, filename):
    output = tmp_path / "tampered"
    prepare_v1_dataset(output, preset="smoke", seed="x", source_rows=_fixtures())
    with (output / filename).open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_v1_directory(output)


def test_validate_checks_manifest_before_trusting_jsonl_paths(tmp_path):
    output = tmp_path / "manifest-path"
    prepare_v1_dataset(output, preset="smoke", seed="x", source_rows=_fixtures())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["q_warm"]["path"] = "../outside.jsonl"
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(payload)
    (output / "manifest.sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="manifest path for q_warm"):
        validate_v1_directory(output)


def test_validate_detects_manifest_tampering(tmp_path):
    output = tmp_path / "manifest-tamper"
    prepare_v1_dataset(output, preset="smoke", seed="x", source_rows=_fixtures())
    with (output / "manifest.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        validate_v1_directory(output)


def test_cli_show_preset_is_offline():
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [sys.executable, "scripts/v1_prepare_data.py", "show-preset", "--preset", "full"],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["q_prompt_groups"] == 2048
    assert payload["q_candidates_at_k4"] == 8192
