from __future__ import annotations

from pathlib import Path

import pytest

from hrm_particle.v1_data import V1Record, write_v1_dataset
from hrm_particle.v1_preflight import audit_prepared_context_lengths


class TinyTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(max(1, len(text.split()))))


def _fixture(tmp_path: Path, *, long_prompt: bool = False) -> dict:
    prompt = ("word " * 100).strip() if long_prompt else "two plus two"

    def record(pool: str, index: int) -> V1Record:
        return V1Record(
            id=f"fixture-{pool}-{index}",
            pool=pool,
            task_type="math",
            source="fixture",
            prompt=prompt + f" case {index}",
            answer="4",
            solution="The answer is 4.",
            verifier="math_equivalence_v1",
            difficulty="easy",
            metadata={},
        )

    write_v1_dataset(
        tmp_path,
        q_warm=[record("q_warm", 1)],
        rl_train=[record("rl_train", 2)],
        eval_records=[record("eval", 3)],
        manifest_fields={"schema_version": 1},
    )
    return {
        "model": {"revision": "pin", "expected_max_position_embeddings": 64},
        "prompting": {
            "condition": "synth,cot",
            "response_prefix": "Solution:",
            "math_suffix": "show work",
            "code_suffix": "return code",
        },
        "generation": {
            "q_collect_max_new_tokens": 8,
            "train_max_new_tokens": 8,
            "math_eval_max_new_tokens": 8,
            "code_eval_max_new_tokens": 8,
        },
    }


def test_token_audit_reports_all_prepared_stages(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    report = audit_prepared_context_lengths(
        tokenizer=TinyTokenizer(), config=config, data_directory=tmp_path
    )
    assert report["max_position_embeddings"] == 64
    assert set(report["stages"]) == {
        "q_warm",
        "rl_train",
        "eval_math",
        "final_q_calibration",
    }
    assert all(value["rows"] == 1 for value in report["stages"].values())


def test_token_audit_optionally_includes_mbpp_code_budget(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["generation"]["code_eval_max_new_tokens"] = 12
    report = audit_prepared_context_lengths(
        tokenizer=TinyTokenizer(),
        config=config,
        data_directory=tmp_path,
        mbpp_prompts=["write a function", "return the sum"],
    )
    assert report["stages"]["eval_mbpp"]["rows"] == 2
    assert report["stages"]["eval_mbpp"]["max_new_tokens"] == 12


def test_token_audit_fails_before_gpu_on_context_overflow(tmp_path: Path) -> None:
    config = _fixture(tmp_path, long_prompt=True)
    with pytest.raises(RuntimeError, match="exceed HRM context"):
        audit_prepared_context_lengths(
            tokenizer=TinyTokenizer(), config=config, data_directory=tmp_path
        )
