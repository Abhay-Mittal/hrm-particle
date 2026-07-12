"""Pinned, decontaminated math/code data preparation for the V1 experiment.

This module is deliberately dependency-light at import time.  Hugging Face
``datasets``/``huggingface_hub`` and EvalPlus are imported only by the default
network loaders.  Every transformation accepts ordinary mappings and iterables,
so schema handling, filtering, selection, and manifests can be tested offline.

Generated code is *never* executed here.  Code records retain already-published
Python tests for a separate sandboxed verifier.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


V1_SCHEMA_VERSION = 1
V1_GENERATOR_VERSION = "hrm_particle_v1_math_code_1"
VALID_POOLS = frozenset({"q_warm", "rl_train", "eval"})
VALID_TASK_TYPES = frozenset({"math", "code"})
DECONTAMINATION_NGRAM_SIZE = 13
GSM_SYMBOLIC_SUBSET_SEED = "hrm-particle-gsm-symbolic-v1"

MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

EASY_CODE_DIFFICULTIES = frozenset(
    {
        "beginner",
        "easy",
        "introductory",
        "interview",
        "medium",
        "level 1",
        "level 2",
        "level 3",
        "1",
        "2",
        "3",
    }
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_LEVEL_RE = re.compile(r"(?:level\s*)?(\d+)", re.IGNORECASE)


class RowRejected(ValueError):
    """A known, counted reason for excluding one upstream row."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True, slots=True)
class V1Record:
    """One normalized prompt group for Q warmup, RL, or evaluation."""

    id: str
    pool: str
    task_type: str
    source: str
    prompt: str
    answer: str | None
    solution: str
    verifier: str
    difficulty: str
    metadata: Mapping[str, Any]
    verification_info: Mapping[str, Any] | None = None
    schema_version: int = V1_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != V1_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={self.schema_version}; expected {V1_SCHEMA_VERSION}"
            )
        if not isinstance(self.id, str) or not self.id or any(char.isspace() for char in self.id):
            raise ValueError("id must be non-empty and contain no whitespace")
        if self.pool not in VALID_POOLS:
            raise ValueError(f"unknown pool {self.pool!r}")
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(f"unknown task_type {self.task_type!r}")
        for name, value in (
            ("source", self.source),
            ("prompt", self.prompt),
            ("solution", self.solution),
            ("verifier", self.verifier),
            ("difficulty", self.difficulty),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            if value != value.strip():
                raise ValueError(f"{name} must not have surrounding whitespace")

        if self.task_type == "math":
            if not isinstance(self.answer, str) or not self.answer.strip():
                raise ValueError("math records require a non-empty answer")
            if self.answer != self.answer.strip():
                raise ValueError("answer must not have surrounding whitespace")
            if self.verifier != "math_equivalence_v1":
                raise ValueError("math records require verifier='math_equivalence_v1'")
            if self.verification_info is not None:
                raise ValueError("math records cannot carry code verification_info")
        else:
            if self.answer is not None:
                raise ValueError("code records use solution, not answer")
            if self.verifier != "python_tests_remote_v1":
                raise ValueError("code records require verifier='python_tests_remote_v1'")
            if self.verification_info is None:
                raise ValueError("code records require verification_info")
            validate_python_verification_info(self.verification_info)

        _validate_finite_json(dict(self.metadata), "metadata")
        if self.verification_info is not None:
            _validate_finite_json(dict(self.verification_info), "verification_info")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "pool": self.pool,
            "task_type": self.task_type,
            "source": self.source,
            "prompt": self.prompt,
            "answer": self.answer,
            "solution": self.solution,
            "verifier": self.verifier,
            "difficulty": self.difficulty,
            "metadata": dict(self.metadata),
            "verification_info": (
                None if self.verification_info is None else dict(self.verification_info)
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V1Record":
        expected = {
            "schema_version",
            "id",
            "pool",
            "task_type",
            "source",
            "prompt",
            "answer",
            "solution",
            "verifier",
            "difficulty",
            "metadata",
            "verification_info",
        }
        missing = expected - set(payload)
        extra = set(payload) - expected
        if missing:
            raise ValueError(f"missing V1Record fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"unexpected V1Record fields: {sorted(extra)}")
        if not isinstance(payload["metadata"], Mapping):
            raise ValueError("metadata must be a mapping")
        verification_info = payload["verification_info"]
        if verification_info is not None and not isinstance(verification_info, Mapping):
            raise ValueError("verification_info must be a mapping or null")
        if not isinstance(payload["schema_version"], int) or isinstance(
            payload["schema_version"], bool
        ):
            raise ValueError("schema_version must be an integer")
        for name in (
            "id",
            "pool",
            "task_type",
            "source",
            "prompt",
            "solution",
            "verifier",
            "difficulty",
        ):
            if not isinstance(payload[name], str):
                raise ValueError(f"{name} must be a string")
        if payload["answer"] is not None and not isinstance(payload["answer"], str):
            raise ValueError("answer must be a string or null")
        record = cls(
            schema_version=payload["schema_version"],
            id=payload["id"],
            pool=payload["pool"],
            task_type=payload["task_type"],
            source=payload["source"],
            prompt=payload["prompt"],
            answer=payload["answer"],
            solution=payload["solution"],
            verifier=payload["verifier"],
            difficulty=payload["difficulty"],
            metadata=dict(payload["metadata"]),
            verification_info=(None if verification_info is None else dict(verification_info)),
        )
        record.validate()
        return record


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    key: str
    repo_id: str
    configs: tuple[str | None, ...]
    split: str
    purpose: str
    license_note: str


SOURCE_SPECS: Mapping[str, DatasetSpec] = {
    "gsm8k_train": DatasetSpec(
        "gsm8k_train", "openai/gsm8k", ("main",), "train", "q_warm", "MIT"
    ),
    "hendrycks_math_train": DatasetSpec(
        "hendrycks_math_train",
        "EleutherAI/hendrycks_math",
        MATH_CONFIGS,
        "train",
        "q_warm",
        "dataset card/source terms apply",
    ),
    "dapo_math_en": DatasetSpec(
        "dapo_math_en",
        "open-r1/DAPO-Math-17k-Processed",
        ("en",),
        "train",
        "q_warm+rl_train",
        "Apache-2.0 upstream",
    ),
    "verified_python": DatasetSpec(
        "verified_python",
        "open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled",
        (None,),
        "train",
        "q_warm+rl_train",
        "mixed upstream sources; inspect dataset card",
    ),
    "gsm8k_eval": DatasetSpec(
        "gsm8k_eval", "openai/gsm8k", ("main",), "test", "eval", "MIT"
    ),
    "math500_eval": DatasetSpec(
        "math500_eval",
        "HuggingFaceH4/MATH-500",
        (None,),
        "test",
        "eval",
        "dataset card/source terms apply",
    ),
    "gsm_symbolic_eval": DatasetSpec(
        "gsm_symbolic_eval",
        "apple/GSM-Symbolic",
        ("p1",),
        "test",
        "eval",
        "CC-BY-NC-ND-4.0; evaluation only",
    ),
}


@dataclass(frozen=True, slots=True)
class V1Preset:
    q_warm: Mapping[str, int]
    rl_train: Mapping[str, int]
    eval: Mapping[str, int | None]

    @property
    def q_prompt_groups(self) -> int:
        return sum(self.q_warm.values())


PRESETS: Mapping[str, V1Preset] = {
    "smoke": V1Preset(
        q_warm={
            "gsm8k_train": 2,
            "hendrycks_math_train": 2,
            "dapo_math_en": 2,
            "verified_python": 2,
        },
        rl_train={"dapo_math_en": 4, "verified_python": 2},
        eval={"gsm8k_eval": 2, "math500_eval": 2, "gsm_symbolic_eval": 2},
    ),
    "pilot": V1Preset(
        q_warm={
            "gsm8k_train": 128,
            "hendrycks_math_train": 128,
            "dapo_math_en": 192,
            "verified_python": 64,
        },
        rl_train={"dapo_math_en": 800, "verified_python": 200},
        eval={"gsm8k_eval": 256, "math500_eval": 256, "gsm_symbolic_eval": 256},
    ),
    # Short V1: keep 8,192 Q candidates, but trim RL storage and evaluation so
    # the complete two-GPU proof-of-concept fits a roughly 12-15 hour window.
    "poc": V1Preset(
        q_warm={
            "gsm8k_train": 512,
            "hendrycks_math_train": 512,
            "dapo_math_en": 512,
            "verified_python": 512,
        },
        rl_train={"dapo_math_en": 2400, "verified_python": 600},
        eval={"gsm8k_eval": 500, "math500_eval": 200, "gsm_symbolic_eval": 200},
    ),
    "full": V1Preset(
        q_warm={
            "gsm8k_train": 512,
            "hendrycks_math_train": 512,
            "dapo_math_en": 512,
            "verified_python": 512,
        },
        rl_train={"dapo_math_en": 4000, "verified_python": 1000},
        eval={"gsm8k_eval": None, "math500_eval": None, "gsm_symbolic_eval": 500},
    ),
}


@dataclass(frozen=True, slots=True)
class NgramIndex:
    n: int
    exact_prompts: frozenset[str]
    ngrams: frozenset[tuple[str, ...]]

    def overlaps(self, prompt: str) -> bool:
        normalized = normalize_prompt(prompt)
        if normalized.casefold() in self.exact_prompts:
            return True
        tokens = tokenize_for_decontamination(normalized)
        if len(tokens) < self.n:
            return False
        return any(tuple(tokens[index : index + self.n]) in self.ngrams for index in range(len(tokens) - self.n + 1))


def _validate_finite_json(value: Any, name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON data") from exc


def _json_copy(value: Any, name: str) -> Any:
    _validate_finite_json(value, name)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def normalize_prompt(value: str) -> str:
    if not isinstance(value, str):
        raise RowRejected("missing_prompt", "prompt must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.strip()
    if not normalized:
        raise RowRejected("missing_prompt")
    return normalized


def _required_text(row: Mapping[str, Any], *names: str, reason: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RowRejected(reason)


def _stable_record_id(source: str, prompt: str, upstream_id: Any = None) -> str:
    payload = f"{source}\0{upstream_id!s}\0{normalize_prompt(prompt)}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    source_slug = re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-")
    return f"v1-{source_slug}-{digest}"


def _last_boxed_payload(text: str) -> str | None:
    for marker in (r"\boxed{", r"\fbox{"):
        starts: list[int] = []
        cursor = 0
        while True:
            found = text.find(marker, cursor)
            if found < 0:
                break
            starts.append(found + len(marker))
            cursor = found + len(marker)
        for start in reversed(starts):
            depth = 1
            for index in range(start, len(text)):
                char = text[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        answer = text[start:index].strip()
                        if answer:
                            return answer
    return None


def _math_answer(answer_or_solution: str) -> str:
    hash_answers = _HASH_ANSWER_RE.findall(answer_or_solution)
    if hash_answers:
        answer = hash_answers[-1].strip().rstrip(".")
        if answer:
            return answer
    boxed = _last_boxed_payload(answer_or_solution)
    if boxed:
        return boxed
    answer = answer_or_solution.strip()
    if not answer:
        raise RowRejected("missing_answer")
    return answer


def _math_record(
    *,
    pool: str,
    source: str,
    prompt: str,
    answer: str,
    solution: str,
    difficulty: str,
    upstream_id: Any,
    metadata: Mapping[str, Any],
) -> V1Record:
    record = V1Record(
        id=_stable_record_id(source, prompt, upstream_id),
        pool=pool,
        task_type="math",
        source=source,
        prompt=normalize_prompt(prompt),
        answer=answer.strip(),
        solution=solution.strip(),
        verifier="math_equivalence_v1",
        difficulty=difficulty.strip(),
        metadata=_json_copy(dict(metadata), "metadata"),
    )
    record.validate()
    return record


def normalize_gsm8k_row(
    row: Mapping[str, Any], *, pool: str, source: str = "gsm8k"
) -> V1Record:
    prompt = _required_text(row, "question", "prompt", reason="missing_prompt")
    solution = _required_text(row, "answer", "solution", reason="missing_solution")
    answer = _math_answer(solution)
    upstream_id = row.get("id", row.get("idx", row.get("index", "")))
    return _math_record(
        pool=pool,
        source=source,
        prompt=prompt,
        answer=answer,
        solution=solution,
        difficulty="grade_school",
        upstream_id=upstream_id,
        metadata={"upstream_id": str(upstream_id)},
    )


def _parse_math_level(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        match = _LEVEL_RE.fullmatch(value.strip())
        if match:
            return int(match.group(1))
    raise RowRejected("missing_difficulty")


def normalize_hendrycks_math_row(
    row: Mapping[str, Any], *, pool: str = "q_warm"
) -> V1Record:
    level = _parse_math_level(row.get("level"))
    if level > 3:
        raise RowRejected("hard_difficulty")
    prompt = _required_text(row, "problem", "prompt", reason="missing_prompt")
    solution = _required_text(row, "solution", reason="missing_solution")
    answer_raw = row.get("answer")
    answer = (
        answer_raw.strip()
        if isinstance(answer_raw, str) and answer_raw.strip()
        else _last_boxed_payload(solution)
    )
    if not answer:
        raise RowRejected("missing_answer")
    subject = str(row.get("type", row.get("subject", "unknown"))).strip() or "unknown"
    upstream_id = row.get("id", row.get("unique_id", ""))
    return _math_record(
        pool=pool,
        source="hendrycks_math_train",
        prompt=prompt,
        answer=answer,
        solution=solution,
        difficulty=f"level_{level}",
        upstream_id=upstream_id,
        metadata={"level": level, "subject": subject, "upstream_id": str(upstream_id)},
    )


def normalize_dapo_row(row: Mapping[str, Any], *, pool: str = "q_warm") -> V1Record:
    prompt = _required_text(row, "prompt", reason="missing_prompt")
    reward_model = row.get("reward_model")
    ground_truth = None
    reward_style = None
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth")
        reward_style = reward_model.get("style")
    solution_raw = row.get("solution", ground_truth)
    if not isinstance(solution_raw, str) or not solution_raw.strip():
        raise RowRejected("missing_answer")
    answer_raw = ground_truth if isinstance(ground_truth, str) and ground_truth.strip() else solution_raw
    extra_info = row.get("extra_info")
    upstream_id = extra_info.get("index", "") if isinstance(extra_info, Mapping) else ""
    return _math_record(
        pool=pool,
        source="dapo_math_en",
        prompt=prompt,
        answer=_math_answer(answer_raw),
        solution=solution_raw,
        difficulty="competition_mixed",
        upstream_id=upstream_id,
        metadata={
            "ability": str(row.get("ability", "MATH")),
            "data_source": str(row.get("data_source", "math_dapo")),
            "reward_style": None if reward_style is None else str(reward_style),
            "upstream_id": str(upstream_id),
        },
    )


def _normalized_difficulty(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        raise RowRejected("missing_difficulty")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RowRejected("missing_difficulty")
        normalized = str(int(value)) if float(value).is_integer() else str(value)
    else:
        normalized = str(value).strip().casefold().replace("_", "-")
        normalized = re.sub(r"\s+", " ", normalized).replace("-", " ")
    if normalized not in EASY_CODE_DIFFICULTIES:
        raise RowRejected("hard_difficulty")
    return normalized.replace(" ", "_")


def validate_python_verification_info(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate published tests structurally, without importing or executing code."""

    if not isinstance(value, Mapping):
        raise ValueError("verification_info must be a mapping")
    language = value.get("language")
    if not isinstance(language, str) or language.casefold() != "python":
        raise ValueError("verification_info.language must be 'python'")
    tests = value.get("test_cases")
    if not isinstance(tests, list) or not tests:
        raise ValueError("verification_info.test_cases must be a non-empty list")
    for index, case in enumerate(tests):
        if not isinstance(case, Mapping):
            raise ValueError(f"test_cases[{index}] must be a mapping")
        case_type = case.get("type")
        # V1 deliberately restricts code RL to strict stdin/stdout tasks.  The
        # remote harness below can validate these without importing candidate
        # code into the verifier process.  Call-based tests need a separate,
        # equally strict harness and are excluded rather than guessed at.
        if case_type != "stdin_stdout":
            raise ValueError(
                f"test_cases[{index}] has unsupported type {case_type!r}; "
                "V1 accepts stdin_stdout only"
            )
        for field in ("input", "output"):
            if field not in case or not isinstance(case[field], str):
                raise ValueError(f"test_cases[{index}].{field} must be a string")
        fn_name = case.get("fn_name", value.get("fn_name"))
        if fn_name not in (None, ""):
            raise ValueError(f"test_cases[{index}] stdin_stdout cannot define fn_name")
    return _json_copy(dict(value), "verification_info")


def normalize_verified_python_row(
    row: Mapping[str, Any], *, pool: str = "q_warm"
) -> V1Record:
    reward = row.get("test_reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or float(reward) != 1.0:
        raise RowRejected("not_verified")
    task_type = row.get("task_type")
    if task_type not in (None, "verifiable_code"):
        raise RowRejected("wrong_task_type")
    prompt = _required_text(row, "problem", "problem_statement", reason="missing_prompt")
    solution = _required_text(row, "gold_standard_solution", reason="missing_solution")
    if len(prompt) > 6_000 or len(solution) > 12_000:
        raise RowRejected("too_long")
    metadata_raw = row.get("metadata")
    if not isinstance(metadata_raw, Mapping):
        raise RowRejected("missing_difficulty")
    difficulty = _normalized_difficulty(metadata_raw.get("difficulty"))
    verification_raw = row.get("verification_info")
    try:
        verification_info = validate_python_verification_info(verification_raw)
    except ValueError as exc:
        reason = "non_python" if "language" in str(exc) else "invalid_test_cases"
        raise RowRejected(reason, str(exc)) from exc
    upstream_id = row.get("problem_id", row.get("in_source_id", ""))
    selected_metadata = {
        "difficulty": difficulty,
        "in_source_id": str(row.get("in_source_id", "")),
        "problem_url": metadata_raw.get("problem_url"),
        "source_dataset": str(row.get("source", "unknown")),
        "upstream_id": str(upstream_id),
    }
    record = V1Record(
        id=_stable_record_id("verified_python", prompt, upstream_id),
        pool=pool,
        task_type="code",
        source="verified_python",
        prompt=normalize_prompt(prompt),
        answer=None,
        solution=solution,
        verifier="python_tests_remote_v1",
        difficulty=difficulty,
        metadata=_json_copy(selected_metadata, "metadata"),
        verification_info=verification_info,
    )
    record.validate()
    return record


def normalize_math500_row(row: Mapping[str, Any], *, pool: str = "eval") -> V1Record:
    prompt = _required_text(row, "problem", "prompt", reason="missing_prompt")
    solution = _required_text(row, "solution", reason="missing_solution")
    answer_raw = _required_text(row, "answer", reason="missing_answer")
    level = _parse_math_level(row.get("level"))
    upstream_id = row.get("unique_id", row.get("id", ""))
    return _math_record(
        pool=pool,
        source="math500",
        prompt=prompt,
        answer=answer_raw,
        solution=solution,
        difficulty=f"level_{level}",
        upstream_id=upstream_id,
        metadata={
            "level": level,
            "subject": str(row.get("subject", "unknown")),
            "upstream_id": str(upstream_id),
        },
    )


def normalize_gsm_symbolic_row(row: Mapping[str, Any], *, pool: str = "eval") -> V1Record:
    prompt = _required_text(row, "question", reason="missing_prompt")
    solution = _required_text(row, "answer", reason="missing_solution")
    original_id = row.get("original_id", "")
    instance = row.get("instance", "")
    upstream_id = f"{original_id}:{instance}"
    return _math_record(
        pool=pool,
        source="gsm_symbolic",
        prompt=prompt,
        answer=_math_answer(solution),
        solution=solution,
        difficulty="grade_school_symbolic",
        upstream_id=upstream_id,
        metadata={
            "instance": instance,
            "original_id": original_id,
            "upstream_id": upstream_id,
        },
    )


Normalizer = Callable[..., V1Record]


def normalize_rows(
    rows: Iterable[Mapping[str, Any]], normalizer: Normalizer, *, pool: str
) -> tuple[list[V1Record], dict[str, int]]:
    """Normalize fixture or upstream rows and return transparent rejection counts."""

    records: list[V1Record] = []
    stats: Counter[str] = Counter()
    for row in rows:
        stats["rows_seen"] += 1
        if not isinstance(row, Mapping):
            stats["invalid_row"] += 1
            continue
        try:
            record = normalizer(row, pool=pool)
        except RowRejected as exc:
            stats[exc.reason] += 1
            continue
        records.append(record)
        stats["accepted"] += 1
    return records, dict(sorted(stats.items()))


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).casefold().encode("utf-8")).hexdigest()


def deduplicate_records(
    records: Iterable[V1Record], *, seen: set[str] | None = None
) -> tuple[list[V1Record], int]:
    seen_keys = seen if seen is not None else set()
    unique: list[V1Record] = []
    duplicates = 0
    for record in records:
        key = prompt_fingerprint(record.prompt)
        if key in seen_keys:
            duplicates += 1
            continue
        seen_keys.add(key)
        unique.append(record)
    return unique, duplicates


def tokenize_for_decontamination(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def build_ngram_index(
    eval_prompts: Iterable[str], *, n: int = DECONTAMINATION_NGRAM_SIZE
) -> NgramIndex:
    if n <= 0:
        raise ValueError("n must be positive")
    exact: set[str] = set()
    ngrams: set[tuple[str, ...]] = set()
    for prompt in eval_prompts:
        normalized = normalize_prompt(prompt)
        exact.add(normalized.casefold())
        tokens = tokenize_for_decontamination(normalized)
        ngrams.update(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return NgramIndex(n=n, exact_prompts=frozenset(exact), ngrams=frozenset(ngrams))


def _stable_order(records: Iterable[V1Record], seed: str) -> list[V1Record]:
    def rank(record: V1Record) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}\0{record.id}".encode("utf-8")).hexdigest()
        return digest, record.id

    return sorted(records, key=rank)


def select_records(
    records: Iterable[V1Record],
    *,
    count: int,
    seed: str,
    pool: str,
    excluded_prompt_keys: set[str] | None = None,
    contamination_index: NgramIndex | None = None,
) -> tuple[list[V1Record], dict[str, int]]:
    """Deterministically take an exact quota after dedupe and decontamination."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if pool not in VALID_POOLS:
        raise ValueError(f"unknown pool {pool!r}")
    if count == 0:
        return [], {"considered": 0, "selected": 0}
    excluded = excluded_prompt_keys if excluded_prompt_keys is not None else set()
    selected: list[V1Record] = []
    stats: Counter[str] = Counter()
    local_seen: set[str] = set()
    for record in _stable_order(records, seed):
        stats["considered"] += 1
        key = prompt_fingerprint(record.prompt)
        if key in excluded or key in local_seen:
            stats["duplicate_or_reserved"] += 1
            continue
        local_seen.add(key)
        if contamination_index is not None and contamination_index.overlaps(record.prompt):
            stats["decontaminated_13gram"] += 1
            continue
        selected.append(replace(record, pool=pool))
        excluded.add(key)
        if len(selected) == count:
            break
    stats["selected"] = len(selected)
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} clean unique records available for {pool}; requested {count}; "
            f"stats={dict(stats)}"
        )
    return selected, dict(sorted(stats.items()))


def resolve_dataset_revision(repo_id: str, *, token: str | None = None) -> str:
    """Resolve ``main`` to an immutable Hub commit SHA (lazy optional import)."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "huggingface_hub is required to resolve dataset revisions; "
            "install requirements-data.txt"
        ) from exc
    info = HfApi(token=token).dataset_info(repo_id, revision="main")
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not revision.strip():
        raise RuntimeError(f"Hub did not return a commit SHA for {repo_id}")
    return revision.strip()


def load_hf_source_rows(
    spec: DatasetSpec, revision: str, *, token: str | None = None
) -> Iterable[Mapping[str, Any]]:
    """Stream all configs for one pinned source (lazy optional import)."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("datasets is required; install requirements-data.txt") from exc
    for config in spec.configs:
        dataset = load_dataset(
            spec.repo_id,
            name=config,
            split=spec.split,
            revision=revision,
            streaming=True,
            token=token,
        )
        yield from dataset


def load_evalplus_mbpp_prompts() -> tuple[list[str], str]:
    """Load MBPP+ prompts only; no candidate code or tests are executed."""

    try:
        from evalplus.data import get_mbpp_plus
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "EvalPlus is optional; install evalplus or omit "
            "--include-mbpp-plus-contamination"
        ) from exc
    problems = get_mbpp_plus()
    if not isinstance(problems, Mapping):
        raise RuntimeError("EvalPlus get_mbpp_plus() returned an unexpected schema")
    prompts: list[str] = []
    for problem in problems.values():
        if isinstance(problem, Mapping):
            prompt = problem.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(normalize_prompt(prompt))
    if not prompts:
        raise RuntimeError("EvalPlus returned no MBPP+ prompts")
    try:
        version = importlib.metadata.version("evalplus")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - defensive
        version = "unknown"
    return prompts, version


def _source_prompt(key: str, row: Mapping[str, Any]) -> str | None:
    names = {
        "gsm8k_eval": ("question", "prompt"),
        "math500_eval": ("problem", "prompt"),
        "gsm_symbolic_eval": ("question",),
    }[key]
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return normalize_prompt(value)
    return None


def _normalizer_for(key: str) -> Normalizer:
    return {
        "gsm8k_train": normalize_gsm8k_row,
        "hendrycks_math_train": normalize_hendrycks_math_row,
        "dapo_math_en": normalize_dapo_row,
        "verified_python": normalize_verified_python_row,
        "gsm8k_eval": normalize_gsm8k_row,
        "math500_eval": normalize_math500_row,
        "gsm_symbolic_eval": normalize_gsm_symbolic_row,
    }[key]


def _resolve_revisions(
    *,
    source_rows: Mapping[str, Iterable[Mapping[str, Any]]] | None,
    supplied: Mapping[str, str] | None,
    resolver: Callable[[str], str] | None,
    token: str | None,
) -> dict[str, str]:
    revisions: dict[str, str] = {}
    supplied_values = dict(supplied or {})
    for repo_id in sorted({spec.repo_id for spec in SOURCE_SPECS.values()}):
        value = supplied_values.get(repo_id)
        if value is None:
            for key, spec in SOURCE_SPECS.items():
                if spec.repo_id == repo_id and key in supplied_values:
                    value = supplied_values[key]
                    break
        if value is None and source_rows is not None:
            value = "offline-fixture"
        if value is None:
            value = resolver(repo_id) if resolver is not None else resolve_dataset_revision(repo_id, token=token)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid resolved revision for {repo_id}")
        revisions[repo_id] = value.strip()
    return revisions


def _materialize_sources(
    *,
    source_rows: Mapping[str, Iterable[Mapping[str, Any]]] | None,
    revisions: Mapping[str, str],
    loader: Callable[[DatasetSpec, str], Iterable[Mapping[str, Any]]] | None,
    token: str | None,
) -> dict[str, list[Mapping[str, Any]]]:
    materialized: dict[str, list[Mapping[str, Any]]] = {}
    if source_rows is not None:
        missing = set(SOURCE_SPECS) - set(source_rows)
        extra = set(source_rows) - set(SOURCE_SPECS)
        if missing:
            raise ValueError(f"offline source_rows missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"offline source_rows has unknown keys: {sorted(extra)}")
    for key, spec in SOURCE_SPECS.items():
        if source_rows is not None:
            rows = source_rows[key]
        elif loader is not None:
            rows = loader(spec, revisions[spec.repo_id])
        else:
            rows = load_hf_source_rows(spec, revisions[spec.repo_id], token=token)
        materialized[key] = list(rows)
    return materialized


def _select_eval_records(
    normalized: Mapping[str, list[V1Record]], preset: V1Preset
) -> tuple[list[V1Record], dict[str, dict[str, int]]]:
    selected: list[V1Record] = []
    seen: set[str] = set()
    stats: dict[str, dict[str, int]] = {}
    for key, target in preset.eval.items():
        seed = GSM_SYMBOLIC_SUBSET_SEED if key == "gsm_symbolic_eval" else f"eval-v1:{key}"
        if key == "gsm_symbolic_eval" and target == 500:
            # The published split contains 100 templates × 50 perturbations.
            # Use five fixed instances per template instead of an unbalanced
            # flat random sample, so uncertainty can later be clustered by the
            # original template if desired.
            balanced = [
                record
                for record in normalized[key]
                if int(record.metadata.get("instance", -1)) < 5
            ]
            ordered = sorted(
                balanced,
                key=lambda record: (
                    str(record.metadata.get("original_id", "")),
                    int(record.metadata.get("instance", -1)),
                ),
            )
        else:
            ordered = _stable_order(normalized[key], seed)
        kept: list[V1Record] = []
        duplicates = 0
        if target == 0:
            stats[key] = {"selected": 0, "duplicate": 0}
            continue
        for record in ordered:
            fingerprint = prompt_fingerprint(record.prompt)
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            kept.append(replace(record, pool="eval"))
            if target is not None and len(kept) == target:
                break
        if target is not None and len(kept) != target:
            raise ValueError(
                f"only {len(kept)} unique eval records available for {key}; requested {target}"
            )
        selected.extend(kept)
        stats[key] = {"selected": len(kept), "duplicate": duplicates}
        if key == "gsm_symbolic_eval" and target == 500:
            instance_counts = Counter(int(record.metadata["instance"]) for record in kept)
            expected_counts = {instance: 100 for instance in range(5)}
            if dict(sorted(instance_counts.items())) != expected_counts:
                raise ValueError(
                    "GSM-Symbolic 500 subset must contain exactly 100 rows for each "
                    f"instance 0..4; got {dict(sorted(instance_counts.items()))}"
                )
            template_counts = Counter(str(record.metadata["original_id"]) for record in kept)
            if len(template_counts) != 100 or set(template_counts.values()) != {5}:
                raise ValueError(
                    "GSM-Symbolic 500 subset must contain five instances for each of "
                    f"100 original templates; got {len(template_counts)} templates"
                )
            stats[key].update(
                {f"instance_{instance}": instance_counts[instance] for instance in range(5)}
            )
            stats[key]["original_templates"] = len(template_counts)
    return selected, stats


def _atomic_write_jsonl(path: Path, records: Sequence[V1Record]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            line = (
                json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode("utf-8")
            handle.write(line)
            digest.update(line)
            byte_count += len(line)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": path.name,
        "records": len(records),
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_v1_dataset(
    output_dir: str | os.PathLike[str],
    *,
    q_warm: Sequence[V1Record],
    rl_train: Sequence[V1Record],
    eval_records: Sequence[V1Record],
    manifest_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically write the three JSONLs, manifest, and manifest checksum sidecar."""

    output = Path(output_dir).expanduser().resolve()
    files = {
        "q_warm": _atomic_write_jsonl(output / "q_warm.jsonl", q_warm),
        "rl_train": _atomic_write_jsonl(output / "rl_train.jsonl", rl_train),
        "eval": _atomic_write_jsonl(output / "eval_math.jsonl", eval_records),
    }
    manifest = {**dict(manifest_fields), "files": files}
    _validate_finite_json(manifest, "manifest")
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    manifest_path = output / "manifest.json"
    _atomic_write_bytes(manifest_path, payload)
    manifest_sha256 = hashlib.sha256(payload).hexdigest()
    _atomic_write_bytes(
        output / "manifest.sha256", f"{manifest_sha256}  manifest.json\n".encode("ascii")
    )
    return {**manifest, "manifest_sha256": manifest_sha256, "output_dir": str(output)}


def prepare_v1_dataset(
    output_dir: str | os.PathLike[str],
    *,
    preset: str = "full",
    seed: str = "1729-v1",
    include_mbpp_plus_contamination: bool = False,
    source_rows: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    revisions: Mapping[str, str] | None = None,
    revision_resolver: Callable[[str], str] | None = None,
    dataset_loader: Callable[[DatasetSpec, str], Iterable[Mapping[str, Any]]] | None = None,
    mbpp_prompt_loader: Callable[[], tuple[list[str], str]] | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Build one reproducible V1 data directory.

    ``source_rows`` is the offline seam: when provided, all seven named source
    iterables are consumed directly and no optional dependency or network call is
    made.  Production runs omit it, resolve immutable Hub revisions, then stream
    each dataset at that exact revision.
    """

    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")
    selected_preset = PRESETS[preset]
    token = hf_token if hf_token is not None else os.getenv("HF_TOKEN")
    resolved_revisions = _resolve_revisions(
        source_rows=source_rows,
        supplied=revisions,
        resolver=revision_resolver,
        token=token,
    )
    raw = _materialize_sources(
        source_rows=source_rows,
        revisions=resolved_revisions,
        loader=dataset_loader,
        token=token,
    )

    normalized: dict[str, list[V1Record]] = {}
    normalization_stats: dict[str, dict[str, int]] = {}
    for key in SOURCE_SPECS:
        pool = "eval" if SOURCE_SPECS[key].purpose == "eval" else "q_warm"
        records, stats = normalize_rows(raw[key], _normalizer_for(key), pool=pool)
        unique, duplicates = deduplicate_records(records)
        stats["exact_duplicates"] = duplicates
        normalized[key] = unique
        normalization_stats[key] = stats

    # Build the leakage index from every raw prompt in every declared evaluation
    # source, not merely the smaller eval subset written by smoke/pilot presets.
    eval_prompts: list[str] = []
    for key in ("gsm8k_eval", "math500_eval", "gsm_symbolic_eval"):
        for row in raw[key]:
            prompt = _source_prompt(key, row)
            if prompt is not None:
                eval_prompts.append(prompt)
    mbpp_version: str | None = None
    mbpp_count = 0
    if include_mbpp_plus_contamination:
        loader = mbpp_prompt_loader or load_evalplus_mbpp_prompts
        mbpp_prompts, mbpp_version = loader()
        eval_prompts.extend(normalize_prompt(prompt) for prompt in mbpp_prompts)
        mbpp_count = len(mbpp_prompts)
    contamination_index = build_ngram_index(eval_prompts)

    eval_records, eval_selection_stats = _select_eval_records(normalized, selected_preset)

    reserved_training: set[str] = set()
    q_warm: list[V1Record] = []
    q_selection_stats: dict[str, dict[str, int]] = {}
    for key, count in selected_preset.q_warm.items():
        chosen, stats = select_records(
            normalized[key],
            count=count,
            seed=f"{seed}:q_warm:{key}",
            pool="q_warm",
            excluded_prompt_keys=reserved_training,
            contamination_index=contamination_index,
        )
        q_warm.extend(chosen)
        q_selection_stats[key] = stats

    rl_train: list[V1Record] = []
    rl_selection_stats: dict[str, dict[str, int]] = {}
    for key, count in selected_preset.rl_train.items():
        chosen, stats = select_records(
            normalized[key],
            count=count,
            seed=f"{seed}:rl_train:{key}",
            pool="rl_train",
            excluded_prompt_keys=reserved_training,
            contamination_index=contamination_index,
        )
        rl_train.extend(chosen)
        rl_selection_stats[key] = stats

    if len(q_warm) != selected_preset.q_prompt_groups:
        raise AssertionError("Q warmup prompt group quota drifted")

    source_manifest = {
        key: {
            "repo_id": spec.repo_id,
            "revision": resolved_revisions[spec.repo_id],
            "configs": list(spec.configs),
            "split": spec.split,
            "purpose": spec.purpose,
            "license_note": spec.license_note,
        }
        for key, spec in SOURCE_SPECS.items()
    }
    manifest_fields = {
        "schema_version": V1_SCHEMA_VERSION,
        "generator_version": V1_GENERATOR_VERSION,
        "preset": preset,
        "selection_seed": seed,
        "sources": source_manifest,
        "planned_counts": {
            "q_warm": dict(selected_preset.q_warm),
            "rl_train": dict(selected_preset.rl_train),
            "eval": dict(selected_preset.eval),
        },
        "q_warm": {
            "prompt_groups": len(q_warm),
            "candidates_at_k4": len(q_warm) * 4,
        },
        "normalization_stats": normalization_stats,
        "selection_stats": {
            "q_warm": q_selection_stats,
            "rl_train": rl_selection_stats,
            "eval": eval_selection_stats,
        },
        "decontamination": {
            "tokenizer": "unicode_nfkc_casefold_words_and_punctuation_v1",
            "ngram_size": contamination_index.n,
            "all_eval_prompts": len(eval_prompts),
            "unique_exact_prompts": len(contamination_index.exact_prompts),
            "unique_ngrams": len(contamination_index.ngrams),
            "includes_mbpp_plus": include_mbpp_plus_contamination,
            "mbpp_plus_prompts": mbpp_count,
            "evalplus_version": mbpp_version,
        },
        "safety": {
            "code_executed_during_preparation": False,
            "required_training_verifier": "remote sandbox (E2B/Morph or equivalent)",
        },
    }
    return write_v1_dataset(
        output_dir,
        q_warm=q_warm,
        rl_train=rl_train,
        eval_records=eval_records,
        manifest_fields=manifest_fields,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_v1_directory(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Transitively verify sidecar -> manifest -> JSONLs and strict schemas."""

    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"V1 data directory does not exist: {directory}")

    manifest_path = directory / "manifest.json"
    sidecar_path = directory / "manifest.sha256"
    if manifest_path.is_symlink() or sidecar_path.is_symlink():
        raise ValueError("manifest files must not be symlinks")
    try:
        sidecar_text = sidecar_path.read_text(encoding="ascii")
        manifest_bytes = manifest_path.read_bytes()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read V1 manifest chain: {exc}") from exc
    sidecar_match = re.fullmatch(r"([0-9a-f]{64})  manifest\.json\n?", sidecar_text)
    if sidecar_match is None:
        raise ValueError("malformed manifest.sha256")
    actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if sidecar_match.group(1) != actual_manifest_hash:
        raise ValueError("manifest checksum mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checksummed manifest.json: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("V1 manifest root must be a mapping")
    if manifest.get("schema_version") != V1_SCHEMA_VERSION:
        raise ValueError("unsupported V1 manifest schema")

    file_manifest = manifest.get("files")
    expected_paths = {
        "q_warm": "q_warm.jsonl",
        "rl_train": "rl_train.jsonl",
        "eval": "eval_math.jsonl",
    }
    if not isinstance(file_manifest, Mapping) or set(file_manifest) != set(expected_paths):
        raise ValueError("manifest must declare exactly q_warm, rl_train, and eval files")

    total = 0
    seen_ids: set[str] = set()
    seen_training_prompts: set[str] = set()
    validated_counts: dict[str, int] = {}
    for pool, expected_name in expected_paths.items():
        info = file_manifest[pool]
        if not isinstance(info, Mapping) or set(info) != {"path", "records", "bytes", "sha256"}:
            raise ValueError(f"invalid file manifest entry {pool!r}")
        if info["path"] != expected_name:
            raise ValueError(f"manifest path for {pool} must be {expected_name!r}")
        if (
            not isinstance(info["records"], int)
            or isinstance(info["records"], bool)
            or info["records"] < 0
            or not isinstance(info["bytes"], int)
            or isinstance(info["bytes"], bool)
            or info["bytes"] < 0
        ):
            raise ValueError(f"manifest counts for {pool} must be non-negative integers")
        if not isinstance(info["sha256"], str) or re.fullmatch(
            r"[0-9a-f]{64}", info["sha256"]
        ) is None:
            raise ValueError(f"manifest SHA256 for {pool} is malformed")

        file_path = directory / expected_name
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"declared JSONL must be a regular file: {expected_name}")
        # Verify the manifest's hash before parsing, then hash the parsed byte
        # stream again so a concurrent replacement cannot evade validation.
        if _sha256_file(file_path) != info["sha256"]:
            raise ValueError(f"checksum mismatch for {expected_name}")
        digest = hashlib.sha256()
        byte_count = 0
        records = 0
        with file_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                digest.update(line)
                byte_count += len(line)
                try:
                    payload = json.loads(line.decode("utf-8"))
                    record = V1Record.from_dict(payload)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{expected_name}:{line_number}: {exc}") from exc
                if record.pool != pool:
                    raise ValueError(f"{expected_name}:{line_number}: pool mismatch")
                if record.id in seen_ids:
                    raise ValueError(f"duplicate record id: {record.id}")
                seen_ids.add(record.id)
                if pool != "eval":
                    fingerprint = prompt_fingerprint(record.prompt)
                    if fingerprint in seen_training_prompts:
                        raise ValueError(f"duplicate training prompt: {record.id}")
                    seen_training_prompts.add(fingerprint)
                records += 1
        if digest.hexdigest() != info["sha256"]:
            raise ValueError(f"checksum changed while validating {expected_name}")
        if records != info["records"] or byte_count != info["bytes"]:
            raise ValueError(f"count/byte mismatch for {expected_name}")
        validated_counts[pool] = records
        total += records
    return {
        "directory": str(directory),
        "manifest_sha256": actual_manifest_hash,
        "records": total,
        "files": validated_counts,
    }
