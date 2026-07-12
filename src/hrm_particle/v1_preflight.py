"""CPU-only preflight checks for prepared V1 prompt context lengths."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .prompting import format_hrm_prompt
from .v1_data import validate_v1_directory
from .v1_utils import sha256_file


def _encode_length(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, list) or not ids:
        raise ValueError("tokenizer returned no token IDs")
    return len(ids)


def _percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object row in {path}")
                rows.append(value)
    return rows


def audit_prepared_context_lengths(
    *,
    tokenizer: Any,
    config: Mapping[str, Any],
    data_directory: Path,
    mbpp_prompts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fail before GPU allocation if any prepared prompt can overflow HRM."""

    data_directory = Path(data_directory).expanduser().resolve()
    manifest = validate_v1_directory(data_directory)
    maximum = int(config["model"]["expected_max_position_embeddings"])
    if maximum <= 0:
        raise ValueError("expected_max_position_embeddings must be positive")
    condition = str(config["prompting"]["condition"])
    response_prefix = str(config["prompting"]["response_prefix"])
    prefix_length = _encode_length(tokenizer, response_prefix)
    stages = {
        "q_warm": (
            "q_warm.jsonl",
            int(config["generation"]["q_collect_max_new_tokens"]),
        ),
        "rl_train": (
            "rl_train.jsonl",
            int(config["generation"]["train_max_new_tokens"]),
        ),
        "eval_math": (
            "eval_math.jsonl",
            int(config["generation"]["math_eval_max_new_tokens"]),
        ),
    }
    reports: dict[str, Any] = {}
    overflows: list[str] = []
    for stage, (filename, max_new_tokens) in stages.items():
        lengths: list[int] = []
        longest: list[tuple[int, str]] = []
        for row in _load_jsonl(data_directory / filename):
            task_type = str(row["task_type"])
            suffix_key = "code_suffix" if task_type == "code" else "math_suffix"
            question = str(row["prompt"]).rstrip() + str(config["prompting"][suffix_key])
            prompt_length = _encode_length(
                tokenizer,
                format_hrm_prompt(question, condition),
            )
            total = prompt_length + prefix_length + max_new_tokens
            lengths.append(total)
            longest.append((total, str(row["id"])))
            if total > maximum:
                overflows.append(
                    f"{stage}:{row['id']} requires {total} tokens (limit {maximum})"
                )
        longest.sort(reverse=True)
        reports[stage] = {
            "rows": len(lengths),
            "max_new_tokens": max_new_tokens,
            "max_total_tokens": max(lengths),
            "p99_total_tokens": _percentile(lengths, 0.99),
            "longest_ids": [identifier for _, identifier in longest[:5]],
        }
    final_q_lengths: list[int] = []
    final_q_longest: list[tuple[int, str]] = []
    for row in _load_jsonl(data_directory / "q_warm.jsonl"):
        task_type = str(row["task_type"])
        suffix_key = "code_suffix" if task_type == "code" else "math_suffix"
        cap_key = (
            "code_eval_max_new_tokens"
            if task_type == "code"
            else "math_eval_max_new_tokens"
        )
        max_new_tokens = int(config["generation"][cap_key])
        question = str(row["prompt"]).rstrip() + str(config["prompting"][suffix_key])
        prompt_length = _encode_length(
            tokenizer,
            format_hrm_prompt(question, condition),
        )
        total = prompt_length + prefix_length + max_new_tokens
        final_q_lengths.append(total)
        final_q_longest.append((total, str(row["id"])))
        if total > maximum:
            overflows.append(
                "final_q_calibration:"
                f"{row['id']} requires {total} tokens (limit {maximum})"
            )
    final_q_longest.sort(reverse=True)
    reports["final_q_calibration"] = {
        "rows": len(final_q_lengths),
        "max_new_tokens": "task-specific math/code evaluation cap",
        "max_total_tokens": max(final_q_lengths),
        "p99_total_tokens": _percentile(final_q_lengths, 0.99),
        "longest_ids": [identifier for _, identifier in final_q_longest[:5]],
    }
    if mbpp_prompts is not None:
        max_new_tokens = int(config["generation"]["code_eval_max_new_tokens"])
        lengths = []
        longest = []
        suffix = str(config["prompting"]["code_suffix"])
        for index, prompt in enumerate(mbpp_prompts):
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"MBPP+ prompt {index} is empty or non-text")
            prompt_length = _encode_length(
                tokenizer,
                format_hrm_prompt(prompt.rstrip() + suffix, condition),
            )
            total = prompt_length + prefix_length + max_new_tokens
            lengths.append(total)
            longest.append((total, f"mbpp-{index}"))
            if total > maximum:
                overflows.append(
                    f"eval_mbpp:mbpp-{index} requires {total} tokens (limit {maximum})"
                )
        if not lengths:
            raise ValueError("mbpp_prompts must not be empty when provided")
        longest.sort(reverse=True)
        reports["eval_mbpp"] = {
            "rows": len(lengths),
            "max_new_tokens": max_new_tokens,
            "max_total_tokens": max(lengths),
            "p99_total_tokens": _percentile(lengths, 0.99),
            "longest_ids": [identifier for _, identifier in longest[:5]],
        }
    if overflows:
        preview = "; ".join(overflows[:10])
        raise RuntimeError(f"prepared prompts exceed HRM context length: {preview}")
    return {
        "format": "hrm-particle-v1-token-audit",
        "model_revision": str(config["model"]["revision"]),
        "max_position_embeddings": maximum,
        "response_prefix_tokens": prefix_length,
        "manifest_sha256": sha256_file(data_directory / "manifest.json"),
        "manifest_validation": manifest,
        "stages": reports,
    }


__all__ = ["audit_prepared_context_lengths"]
