"""Reproducible datasets and exact-answer helpers for the HRM particle POC.

The synthetic generator intentionally keeps training, in-distribution evaluation,
and held-template OOD evaluation in separate JSONL files.  A record is fully
self-describing and can be verified without executing an expression from disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
GENERATOR_VERSION = "synthetic_exact_arithmetic_v1"
SYNTHETIC_SOURCE = "synthetic_exact_arithmetic"
PUBLIC_SMOKE_EVAL_SEED = "PUBLIC-SMOKE-EVAL-SEED-NOT-FOR-RESULTS"
VALID_SPLITS = frozenset({"train", "dev", "test", "ood", "external_train", "sample"})
EVAL_SPLITS = frozenset({"dev", "test", "ood"})

# These Big-Math families are common benchmark/evaluation sources.  The import
# helper rejects them rather than quietly contaminating a training set.
DEFAULT_BIGMATH_EVAL_SOURCE_DENYLIST = frozenset(
    {
        "aops_forum",
        "amc_aime",
        "gsm8k",
        "harp",
        "math",
        "omni-math",
        "omnimath",
        "olympiads",
    }
)

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_FRACTION_RE = re.compile(r"^([+-]?\d+)\s*/\s*([+-]?\d+)$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)$")
_LATEX_FRACTION_RE = re.compile(
    r"^([+-]?)\\(?:d?frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}$"
)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_PLAIN_NUMBER_RE = re.compile(
    r"(?<![\w.])([+-]?\d+(?:\s*/\s*[+-]?\d+|\.\d+)?(?:\\?%)?)(?!\w)"
)


def _strip_outer_braces(text: str) -> str:
    text = text.strip()
    while len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        depth = 0
        wraps_all = True
        for index, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    wraps_all = False
                    break
            if depth < 0:
                wraps_all = False
                break
        if not wraps_all or depth != 0:
            break
        text = text[1:-1].strip()
    return text


def _extract_boxed(text: str) -> str | None:
    """Return the last balanced ``\\boxed{...}`` payload, if present."""

    marker = r"\boxed{"
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
                    return text[start:index].strip()
    return None


def canonicalize_exact_answer(value: str | int | Fraction) -> str:
    """Canonicalize an integer, rational, finite decimal, or simple LaTeX fraction.

    The result is either ``"n"`` or ``"n/d"`` with a positive denominator.
    Arbitrary Python/SymPy expressions are deliberately not evaluated.
    """

    if isinstance(value, Fraction):
        fraction = value
    elif isinstance(value, int) and not isinstance(value, bool):
        fraction = Fraction(value)
    else:
        text = str(value).strip()
        text = text.replace("\u2212", "-").replace("\u2013", "-")
        text = text.rstrip(".,; ")
        text = _strip_outer_braces(text)
        if text.startswith("$") and text.endswith("$") and len(text) >= 2:
            text = text[1:-1].strip()
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1].rstrip()
            if text.endswith("\\"):
                text = text[:-1].rstrip()
        text = text.replace(",", "").strip()

        latex_match = _LATEX_FRACTION_RE.fullmatch(text)
        fraction_match = _FRACTION_RE.fullmatch(text)
        if latex_match:
            sign, numerator, denominator = latex_match.groups()
            numerator_value = int(numerator)
            if sign == "-":
                numerator_value = -numerator_value
            denominator_value = int(denominator)
            if denominator_value == 0:
                raise ValueError("zero denominator")
            fraction = Fraction(numerator_value, denominator_value)
        elif fraction_match:
            numerator, denominator = fraction_match.groups()
            if int(denominator) == 0:
                raise ValueError("zero denominator")
            fraction = Fraction(int(numerator), int(denominator))
        elif _INTEGER_RE.fullmatch(text):
            fraction = Fraction(int(text))
        elif _DECIMAL_RE.fullmatch(text):
            fraction = Fraction(text)
        else:
            raise ValueError(f"not a supported exact numeric answer: {value!r}")
        if is_percent:
            fraction /= 100

    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def extract_final_answer(text: str) -> str | None:
    """Extract and canonicalize the last explicitly presented numeric answer."""

    candidates: list[str] = []
    candidates.extend(_ANSWER_TAG_RE.findall(text))
    boxed = _extract_boxed(text)
    if boxed is not None:
        candidates.append(boxed)
    candidates.extend(_HASH_ANSWER_RE.findall(text))
    if not candidates:
        candidates.extend(match.group(1) for match in _PLAIN_NUMBER_RE.finditer(text))

    for candidate in reversed(candidates):
        try:
            return canonicalize_exact_answer(candidate)
        except ValueError:
            continue
    return None


def verify_exact_answer(response: str, reference: str | int | Fraction) -> bool:
    extracted = extract_final_answer(response)
    if extracted is None:
        return False
    try:
        return extracted == canonicalize_exact_answer(reference)
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class MathRecord:
    """One JSONL example in the POC's stable schema."""

    id: str
    split: str
    family: str
    template_id: str
    prompt: str
    answer: str
    answer_type: str
    verifier: str
    source: str
    difficulty: int
    metadata: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not self.id or any(char.isspace() for char in self.id):
            raise ValueError("record id must be non-empty and contain no whitespace")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"unknown split {self.split!r}")
        for name, value in (
            ("family", self.family),
            ("template_id", self.template_id),
            ("prompt", self.prompt),
            ("answer", self.answer),
            ("answer_type", self.answer_type),
            ("verifier", self.verifier),
            ("source", self.source),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not 1 <= self.difficulty <= 5:
            raise ValueError("difficulty must be in [1, 5]")
        if self.verifier == "exact_fraction_v1":
            canonical = canonicalize_exact_answer(self.answer)
            if canonical != self.answer:
                raise ValueError(f"answer is not canonical: {self.answer!r} != {canonical!r}")
        try:
            json.dumps(dict(self.metadata), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be finite JSON data") from exc

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "split": self.split,
            "family": self.family,
            "template_id": self.template_id,
            "prompt": self.prompt,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "verifier": self.verifier,
            "source": self.source,
            "difficulty": self.difficulty,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MathRecord":
        required = {
            "schema_version",
            "id",
            "split",
            "family",
            "template_id",
            "prompt",
            "answer",
            "answer_type",
            "verifier",
            "source",
            "difficulty",
            "metadata",
        }
        missing = required - payload.keys()
        extra = payload.keys() - required
        if missing:
            raise ValueError(f"missing record fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"unexpected record fields: {sorted(extra)}")
        if isinstance(payload["schema_version"], bool) or not isinstance(
            payload["schema_version"], int
        ):
            raise ValueError("schema_version must be an integer")
        if isinstance(payload["difficulty"], bool) or not isinstance(payload["difficulty"], int):
            raise ValueError("difficulty must be an integer")
        string_fields = (
            "id",
            "split",
            "family",
            "template_id",
            "prompt",
            "answer",
            "answer_type",
            "verifier",
            "source",
        )
        wrong_string_types = [name for name in string_fields if not isinstance(payload[name], str)]
        if wrong_string_types:
            raise ValueError(f"record fields must be strings: {wrong_string_types}")
        metadata = payload["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        record = cls(
            schema_version=payload["schema_version"],
            id=payload["id"],
            split=payload["split"],
            family=payload["family"],
            template_id=payload["template_id"],
            prompt=payload["prompt"],
            answer=payload["answer"],
            answer_type=payload["answer_type"],
            verifier=payload["verifier"],
            source=payload["source"],
            difficulty=payload["difficulty"],
            metadata=dict(metadata),
        )
        record.validate()
        return record


def _answer_type(answer: Fraction) -> str:
    return "integer" if answer.denominator == 1 else "rational"


def _magnitude(difficulty: int) -> int:
    return (10, 40, 200, 2_000, 20_000)[difficulty - 1]


def _nonzero(rng: random.Random, low: int, high: int) -> int:
    value = 0
    while value == 0:
        value = rng.randint(low, high)
    return value


def _signed(rng: random.Random, magnitude: int) -> int:
    return _nonzero(rng, -magnitude, magnitude)


def _make_problem(
    family: str,
    difficulty: int,
    rng: random.Random,
) -> tuple[dict[str, int], Fraction, str]:
    """Generate operands, answer, and a non-executable semantic signature."""

    magnitude = _magnitude(difficulty)
    if family == "nested_arithmetic":
        a = _signed(rng, magnitude)
        b = _signed(rng, magnitude)
        c = _nonzero(rng, 2, max(3, min(97, magnitude)))
        d = _signed(rng, magnitude * max(1, difficulty))
        answer = Fraction((a + b) * c - d)
        operands = {"a": a, "b": b, "c": c, "d": d}
        signature = f"({a}+{b})*{c}-{d}"
    elif family == "affine_equation":
        a = _nonzero(rng, -max(3, magnitude // 2), max(3, magnitude // 2))
        x = _signed(rng, magnitude)
        b = _signed(rng, magnitude * 2)
        c = a * x + b
        answer = Fraction(x)
        operands = {"a": a, "b": b, "c": c}
        signature = f"{a}*x+{b}={c}"
    elif family == "rational_sum":
        denominator_limit = (9, 20, 50, 120, 250)[difficulty - 1]
        b = rng.randint(2, denominator_limit)
        d = rng.randint(2, denominator_limit)
        f = rng.randint(2, denominator_limit)
        a = _signed(rng, denominator_limit * 2)
        c = _signed(rng, denominator_limit * 2)
        e = _signed(rng, denominator_limit * 2)
        answer = Fraction(a, b) + Fraction(c, d) - Fraction(e, f)
        operands = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}
        signature = f"{a}/{b}+{c}/{d}-{e}/{f}"
    elif family == "inventory_value":
        initial = rng.randint(magnitude, magnitude * 5)
        delivered = rng.randint(magnitude, magnitude * 4)
        removed = rng.randint(0, initial + delivered)
        unit_price = rng.randint(2, max(3, min(500, magnitude)))
        answer = Fraction((initial + delivered - removed) * unit_price)
        operands = {
            "initial": initial,
            "delivered": delivered,
            "removed": removed,
            "unit_price": unit_price,
        }
        signature = f"({initial}+{delivered}-{removed})*{unit_price}"
    elif family == "ratio_share":
        left = rng.randint(1, max(3, min(30, magnitude)))
        right = rng.randint(1, max(3, min(30, magnitude)))
        scale = rng.randint(magnitude, magnitude * max(2, difficulty + 1))
        total = (left + right) * scale
        answer = Fraction(right * scale)
        operands = {"left": left, "right": right, "total": total}
        signature = f"ratio:{left}:{right}:total={total}:right"
    elif family == "reverse_percent":
        # Percent is selected so the original amount is integral and exact.
        percent = rng.choice((5, 10, 12, 15, 20, 25, 40, 50, 60, 75))
        base = rng.randint(magnitude, magnitude * max(2, difficulty + 1))
        final = base * (100 + percent)
        displayed = Fraction(final, 100)
        answer = Fraction(base)
        operands = {
            "percent": percent,
            "displayed_numerator": displayed.numerator,
            "displayed_denominator": displayed.denominator,
        }
        signature = f"increase:{percent}:final={displayed}:base"
    else:
        raise ValueError(f"unknown synthetic family {family!r}")
    return operands, answer, signature


IN_DOMAIN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "nested_arithmetic": (
        "Compute exactly: ({a} + ({b})) * {c} - ({d}).",
        "Add {a} and ({b}), multiply that sum by {c}, then subtract ({d}).",
    ),
    "affine_equation": (
        "Solve for x: ({a})x + ({b}) = {c}.",
        "Find the exact value of x satisfying ({a}) * x + ({b}) = {c}.",
    ),
    "rational_sum": (
        "Reduce to lowest terms: ({a}/{b}) + ({c}/{d}) - ({e}/{f}).",
        "What is the exact rational value of ({a}/{b}) + ({c}/{d}) - ({e}/{f})?",
    ),
    "inventory_value": (
        "A warehouse starts with {initial} parts, receives {delivered}, and ships {removed}. Each remaining part is worth ${unit_price}. What is the total value in dollars?",
        "There are {initial} units initially. After adding {delivered} and removing {removed}, multiply the remainder by {unit_price}. What is the result?",
    ),
    "ratio_share": (
        "An amount of {total} is divided in the ratio {left}:{right}. What is the right-hand share corresponding to {right}?",
        "Two shares are in ratio {left} to {right} and total {total}. Find the second share.",
    ),
    "reverse_percent": (
        "After increasing an original number by {percent}%, the result is {displayed}. What was the original number?",
        "A price rose {percent}% to {displayed}. Find its exact price before the increase.",
    ),
}

OOD_TEMPLATES: dict[str, tuple[str, ...]] = {
    "nested_arithmetic": (
        "Take the sum of {a} and {b}; scale it by {c}; offset the outcome downward by {d}. Report the exact result.",
    ),
    "affine_equation": (
        "A hidden integer, multiplied by {a} and then increased by {b}, becomes {c}. Recover the hidden integer.",
    ),
    "rational_sum": (
        "Combine the signed portions {a}/{b} and {c}/{d}, then remove {e}/{f}. Give one reduced fraction.",
    ),
    "inventory_value": (
        "A vault's count changes from {initial} by +{delivered} and -{removed}. Valuing each survivor at {unit_price}, determine the aggregate value.",
    ),
    "ratio_share": (
        "Split {total} into {left}+{right} equal ratio-units. How much belongs to the block of {right} units?",
    ),
    "reverse_percent": (
        "The displayed amount {displayed} is {percent} percent above an unknown baseline. Determine that baseline exactly.",
    ),
}


def synthetic_template_ids(partition: str) -> frozenset[str]:
    templates = IN_DOMAIN_TEMPLATES if partition == "in_domain" else OOD_TEMPLATES
    if partition not in {"in_domain", "ood"}:
        raise ValueError("partition must be 'in_domain' or 'ood'")
    prefix = "id" if partition == "in_domain" else "ood"
    return frozenset(
        f"{prefix}.{family}.{index}" for family, values in templates.items() for index in range(len(values))
    )


def _format_displayed(operands: Mapping[str, int]) -> dict[str, str | int]:
    values: dict[str, str | int] = dict(operands)
    if "displayed_numerator" in operands:
        value = Fraction(operands["displayed_numerator"], operands["displayed_denominator"])
        values["displayed"] = canonicalize_exact_answer(value)
    return values


def _seed_fingerprint(seed: int | str) -> str:
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:16]


def derive_split_seed(seed: int | str, split: str) -> int:
    digest = hashlib.blake2b(
        f"{GENERATOR_VERSION}\0{split}\0{seed}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def generate_synthetic_records(
    *,
    split: str,
    count: int,
    seed: int | str,
    min_difficulty: int = 2,
    max_difficulty: int = 5,
    excluded_semantic_hashes: set[str] | frozenset[str] | None = None,
    excluded_prompt_hashes: set[str] | frozenset[str] | None = None,
) -> list[MathRecord]:
    """Generate a deterministic, duplicate-free synthetic split."""

    if split not in {"train", "dev", "test", "ood", "sample"}:
        raise ValueError(f"synthetic generation does not support split={split!r}")
    if count < 0:
        raise ValueError("count must be non-negative")
    if not (1 <= min_difficulty <= max_difficulty <= 5):
        raise ValueError("difficulty bounds must satisfy 1 <= min <= max <= 5")

    partition = "ood" if split == "ood" else "in_domain"
    templates = OOD_TEMPLATES if partition == "ood" else IN_DOMAIN_TEMPLATES
    prefix = "ood" if partition == "ood" else "id"
    rng = random.Random(derive_split_seed(seed, split))
    families = tuple(sorted(templates))
    records: list[MathRecord] = []
    seen_signatures: set[str] = set()
    seen_prompts: set[str] = set()
    excluded_semantics = excluded_semantic_hashes or frozenset()
    excluded_prompts = excluded_prompt_hashes or frozenset()
    attempts = 0
    max_attempts = max(1_000, count * 100)

    while len(records) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"could not generate {count} unique {split} records")
        family = families[len(records) % len(families)]
        difficulty = rng.randint(min_difficulty, max_difficulty)
        operands, answer, semantic = _make_problem(family, difficulty, rng)
        semantic_digest = hashlib.sha256(f"{family}\0{semantic}".encode("utf-8")).hexdigest()
        if semantic_digest in seen_signatures or semantic_digest in excluded_semantics:
            continue
        template_index = rng.randrange(len(templates[family]))
        template_id = f"{prefix}.{family}.{template_index}"
        prompt = templates[family][template_index].format(**_format_displayed(operands))
        prompt_digest = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()
        if prompt_digest in seen_prompts or prompt_digest in excluded_prompts:
            continue
        seen_signatures.add(semantic_digest)
        seen_prompts.add(prompt_digest)
        answer_text = canonicalize_exact_answer(answer)
        record_digest = hashlib.sha256(
            f"{GENERATOR_VERSION}\0{split}\0{semantic_digest}\0{template_id}".encode("utf-8")
        ).hexdigest()[:20]
        records.append(
            MathRecord(
                id=f"syn-{split}-{record_digest}",
                split=split,
                family=family,
                template_id=template_id,
                prompt=prompt,
                answer=answer_text,
                answer_type=_answer_type(answer),
                verifier="exact_fraction_v1",
                source=SYNTHETIC_SOURCE,
                difficulty=difficulty,
                metadata={
                    "generator_version": GENERATOR_VERSION,
                    "template_partition": partition,
                    "semantic_hash": semantic_digest,
                    "operands": operands,
                },
            )
        )
    return records


def write_jsonl(path: str | os.PathLike[str], records: Iterable[MathRecord]) -> dict[str, Any]:
    """Atomically write validated JSONL and return its count/checksum."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False, prefix=f".{destination.name}."
    ) as handle:
        temporary = Path(handle.name)
        try:
            for record in records:
                payload = record.to_dict()
                line = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    return {"count": count, "sha256": digest.hexdigest()}


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[MathRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("line must contain a JSON object")
                yield MathRecord.from_dict(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc


def load_jsonl(path: str | os.PathLike[str]) -> list[MathRecord]:
    return list(iter_jsonl(path))


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_isolation(splits: Mapping[str, Sequence[MathRecord]]) -> dict[str, Any]:
    """Reject IDs, prompts, or generated semantic problems shared across splits."""

    all_ids: dict[str, str] = {}
    all_prompts: dict[str, str] = {}
    all_semantics: dict[str, str] = {}
    ood_ids = synthetic_template_ids("ood")
    in_domain_ids = synthetic_template_ids("in_domain")

    for expected_split, records in splits.items():
        for record in records:
            record.validate()
            if record.split != expected_split:
                raise ValueError(
                    f"record {record.id} declares split={record.split!r}, loaded as {expected_split!r}"
                )
            if record.id in all_ids:
                raise ValueError(f"record id overlap between {all_ids[record.id]} and {expected_split}: {record.id}")
            all_ids[record.id] = expected_split
            prompt_hash = hashlib.sha256(record.prompt.strip().encode("utf-8")).hexdigest()
            if prompt_hash in all_prompts:
                raise ValueError(
                    f"prompt overlap between {all_prompts[prompt_hash]} and {expected_split}: {record.id}"
                )
            all_prompts[prompt_hash] = expected_split
            semantic_hash = str(record.metadata.get("semantic_hash", ""))
            if semantic_hash:
                if semantic_hash in all_semantics:
                    raise ValueError(
                        f"semantic overlap between {all_semantics[semantic_hash]} and {expected_split}: {record.id}"
                    )
                all_semantics[semantic_hash] = expected_split

            if record.source == SYNTHETIC_SOURCE:
                if expected_split == "ood" and record.template_id not in ood_ids:
                    raise ValueError(f"OOD record uses non-OOD template: {record.id}")
                if expected_split != "ood" and record.template_id not in in_domain_ids:
                    raise ValueError(f"non-OOD record uses held-out OOD template: {record.id}")
    return {
        "records": len(all_ids),
        "split_counts": {name: len(records) for name, records in splits.items()},
        "unique_prompts": len(all_prompts),
        "unique_semantics": len(all_semantics),
    }


def generate_synthetic_dataset(
    output_dir: str | os.PathLike[str],
    *,
    sizes: Mapping[str, int],
    train_seed: int | str,
    eval_seed: int | str,
    min_difficulty: int = 2,
    max_difficulty: int = 5,
) -> dict[str, Any]:
    """Generate isolated splits and a checksum manifest.

    ``eval_seed`` should be supplied via a private file or environment variable
    for a real experiment.  Only a one-way fingerprint is written to disk.
    """

    unknown = set(sizes) - {"train", "dev", "test", "ood"}
    if unknown:
        raise ValueError(f"unsupported synthetic split names: {sorted(unknown)}")
    if not sizes:
        raise ValueError("at least one split size is required")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated: dict[str, list[MathRecord]] = {}
    used_semantics: set[str] = set()
    used_prompts: set[str] = set()
    # Canonical order makes train the fixed priority set regardless of caller
    # mapping order; later sealed splits deterministically resample collisions.
    for split in ("train", "dev", "test", "ood"):
        if split not in sizes:
            continue
        count = sizes[split]
        split_seed = train_seed if split == "train" else eval_seed
        generated[split] = generate_synthetic_records(
            split=split,
            count=int(count),
            seed=split_seed,
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
            excluded_semantic_hashes=used_semantics,
            excluded_prompt_hashes=used_prompts,
        )
        used_semantics.update(
            str(record.metadata["semantic_hash"]) for record in generated[split]
        )
        used_prompts.update(
            hashlib.sha256(record.prompt.strip().encode("utf-8")).hexdigest()
            for record in generated[split]
        )
    validation = validate_split_isolation(generated)
    files: dict[str, Any] = {}
    for split, records in generated.items():
        filename = f"{split}.jsonl"
        files[split] = {"path": filename, **write_jsonl(output / filename, records)}

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "source": SYNTHETIC_SOURCE,
        "seed_fingerprints": {
            "train": _seed_fingerprint(train_seed),
            "eval": _seed_fingerprint(eval_seed),
        },
        "evaluation_seed_status": (
            "public_smoke_only"
            if str(eval_seed) == PUBLIC_SMOKE_EVAL_SEED
            else "private_user_supplied"
        ),
        "difficulty": {"min": min_difficulty, "max": max_difficulty},
        "files": files,
        "validation": validation,
        "template_partitions": {
            "in_domain": sorted(synthetic_template_ids("in_domain")),
            "ood": sorted(synthetic_template_ids("ood")),
        },
    }
    manifest_path = output / "manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output, delete=False, prefix=".manifest."
    ) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)
    return manifest


def validate_dataset_directory(path: str | os.PathLike[str]) -> dict[str, Any]:
    directory = Path(path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifest schema version mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("manifest has no files")
    splits: dict[str, list[MathRecord]] = {}
    for split, entry in files.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"invalid manifest file entry for {split}")
        file_path = directory / str(entry["path"])
        if not file_path.is_file():
            raise FileNotFoundError(f"missing split file: {file_path}")
        actual_checksum = file_sha256(file_path)
        if actual_checksum != entry.get("sha256"):
            raise ValueError(f"checksum mismatch for {file_path}")
        records = load_jsonl(file_path)
        if len(records) != int(entry.get("count", -1)):
            raise ValueError(f"record count mismatch for {file_path}")
        splits[str(split)] = records
    return {"manifest": manifest, "isolation": validate_split_isolation(splits)}


def _normalize_source_name(source: Any) -> str:
    return str(source).strip().lower().replace("_", "-")


def convert_big_math_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_sources: Sequence[str],
    limit: int,
    seed: int | str,
    upstream_dataset: str = "SynthLabsAI/Big-Math-RL-Verified",
    min_solve_rate: float | None = None,
    max_solve_rate: float | None = None,
    max_rows_scanned: int = 100_000,
    require_limit: bool = False,
) -> tuple[list[MathRecord], dict[str, int]]:
    """Convert an explicitly allow-listed Big-Math subset to exact-rational train rows.

    Evaluation-family sources are always rejected.  Answers outside the POC's
    strict numeric verifier are skipped, never guessed or symbolically executed.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_rows_scanned <= 0:
        raise ValueError("max_rows_scanned must be positive")
    if min_solve_rate is not None and not 0 <= min_solve_rate <= 1:
        raise ValueError("min_solve_rate must be in [0, 1]")
    if max_solve_rate is not None and not 0 <= max_solve_rate <= 1:
        raise ValueError("max_solve_rate must be in [0, 1]")
    if (
        min_solve_rate is not None
        and max_solve_rate is not None
        and min_solve_rate > max_solve_rate
    ):
        raise ValueError("min_solve_rate cannot exceed max_solve_rate")
    allowed = {_normalize_source_name(source) for source in allowed_sources}
    if not allowed:
        raise ValueError("allowed_sources must be explicitly provided")
    forbidden = allowed & {_normalize_source_name(source) for source in DEFAULT_BIGMATH_EVAL_SOURCE_DENYLIST}
    if forbidden:
        raise ValueError(f"refusing evaluation-family Big-Math sources: {sorted(forbidden)}")

    rng = random.Random(derive_split_seed(seed, "external_train"))
    stats: Counter[str] = Counter()
    # Reservoir sampling keeps memory O(limit) while selecting uniformly from
    # strict-numeric eligible rows within the explicit scan cap. Prompt hashes
    # are O(max_rows_scanned), so even an infinite stream is bounded.
    candidates: list[tuple[str, str, str, str, str, float | None]] = []
    seen_prompts: set[str] = set()
    eligible_seen = 0
    exhausted = False
    iterator = iter(rows)
    for _ in range(max_rows_scanned):
        try:
            row = next(iterator)
        except StopIteration:
            exhausted = True
            break
        stats["rows_scanned"] += 1
        source = _normalize_source_name(row.get("source", ""))
        if source not in allowed:
            stats["source_filtered"] += 1
            continue
        # Raw SynthLabs uses problem/answer; Open-R1's processed mirror uses
        # prompt/solution.  Never guess beyond these documented aliases.
        problem = str(row.get("problem") or row.get("prompt") or "").strip()
        raw_answer = str(row.get("answer") or row.get("solution") or "").strip()
        if not problem or not raw_answer:
            stats["missing_fields"] += 1
            continue
        solve_rate_raw = row.get("llama8b_solve_rate")
        try:
            solve_rate = float(solve_rate_raw) if solve_rate_raw is not None else None
        except (TypeError, ValueError):
            solve_rate = None
        if solve_rate is not None and not math.isfinite(solve_rate):
            solve_rate = None
        if min_solve_rate is not None and (solve_rate is None or solve_rate < min_solve_rate):
            stats["solve_rate_filtered"] += 1
            continue
        if max_solve_rate is not None and (solve_rate is None or solve_rate > max_solve_rate):
            stats["solve_rate_filtered"] += 1
            continue
        try:
            extracted_answer = canonicalize_exact_answer(raw_answer)
        except ValueError:
            # Rich upstream answers are accepted only when they use an explicit
            # final-answer marker.  Falling back to an arbitrary last number
            # would turn answers such as ``x^2`` into the incorrect label 2.
            has_explicit_marker = (
                bool(_ANSWER_TAG_RE.search(raw_answer))
                or r"\boxed{" in raw_answer
                or bool(_HASH_ANSWER_RE.search(raw_answer))
            )
            extracted_answer = extract_final_answer(raw_answer) if has_explicit_marker else None
        if extracted_answer is None:
            stats["non_numeric_answer"] += 1
            continue
        prompt_hash = hashlib.sha256(problem.encode("utf-8")).hexdigest()
        if prompt_hash in seen_prompts:
            stats["duplicate_prompt"] += 1
            continue
        seen_prompts.add(prompt_hash)
        digest = hashlib.sha256(f"{source}\0{problem}\0{extracted_answer}".encode("utf-8")).hexdigest()
        domain = row.get("domain")
        if isinstance(domain, Sequence) and not isinstance(domain, (str, bytes)):
            domain_text = str(domain[-1]) if domain else "unknown"
        else:
            domain_text = str(domain or "unknown")
        candidate = (digest, problem, extracted_answer, source, domain_text, solve_rate)
        eligible_seen += 1
        if len(candidates) < limit:
            candidates.append(candidate)
        else:
            replacement = rng.randrange(eligible_seen)
            if replacement < limit:
                candidates[replacement] = candidate

    stats["max_rows_scanned"] = max_rows_scanned
    stats["scan_cap_reached"] = int(
        not exhausted and stats["rows_scanned"] == max_rows_scanned
    )
    stats["eligible_numeric_rows"] = eligible_seen
    stats["requested"] = limit
    stats["accepted"] = len(candidates)
    if require_limit and len(candidates) < limit:
        raise ValueError(
            f"requested {limit} Big-Math rows but found only {len(candidates)} eligible rows "
            f"after scanning {stats['rows_scanned']} (cap {max_rows_scanned}); loosen filters, "
            "raise --max-rows-scanned, or pass --allow-fewer explicitly"
        )

    accepted: list[MathRecord] = []
    for digest, prompt, answer, source, domain_text, solve_rate in sorted(candidates):
        accepted.append(
            MathRecord(
                id=f"bigmath-{digest[:20]}",
                split="external_train",
                family=domain_text.strip() or "unknown",
                template_id="external.big_math",
                prompt=prompt,
                answer=answer,
                answer_type="integer" if "/" not in answer else "rational",
                verifier="exact_fraction_v1",
                source=f"big_math/{source}",
                difficulty=3,
                metadata={
                    "upstream_dataset": upstream_dataset,
                    "upstream_source": source,
                    "llama8b_solve_rate": solve_rate,
                    "semantic_hash": digest,
                },
            )
        )
    return accepted, dict(stats)
