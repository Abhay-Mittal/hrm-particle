from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from hrm_particle.data import (
    MathRecord,
    canonicalize_exact_answer,
    convert_big_math_rows,
    extract_final_answer,
    generate_synthetic_dataset,
    generate_synthetic_records,
    load_jsonl,
    synthetic_template_ids,
    validate_dataset_directory,
    validate_split_isolation,
    verify_exact_answer,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", "42"),
        ("-6/8", "-3/4"),
        ("0.125", "1/8"),
        (r"\frac{15}{-10}", "-3/2"),
        (Fraction(9, 3), "3"),
        ("{1,250}", "1250"),
        ("12.5%", "1/8"),
        (r"10\%", "1/10"),
    ],
)
def test_canonicalize_exact_answer(raw, expected):
    assert canonicalize_exact_answer(raw) == expected


@pytest.mark.parametrize("raw", ["", "x + 1", "1/0", "inf", "2 ** 8"])
def test_canonicalize_rejects_non_numeric_or_unsafe_values(raw):
    with pytest.raises(ValueError):
        canonicalize_exact_answer(raw)


def test_extract_final_answer_prefers_explicit_last_answer():
    assert extract_final_answer("We tried 12. <answer>-6/8</answer>") == "-3/4"
    assert extract_final_answer(r"Reasoning 17; therefore \boxed{\frac{-6}{8}}") == "-3/4"
    assert extract_final_answer("work 2\n#### 7") == "7"
    assert extract_final_answer("first 2, final 7.") == "7"
    assert extract_final_answer(r"The final discount is 10\%.") == "1/10"
    assert verify_exact_answer("The answer is <answer>0.5</answer>", "1/2")
    assert not verify_exact_answer("I cannot solve it", "1/2")


def test_record_schema_round_trip_is_strict():
    record = generate_synthetic_records(split="sample", count=1, seed=7)[0]
    payload = record.to_dict()
    assert MathRecord.from_dict(payload) == record
    with pytest.raises(ValueError, match="unexpected record fields"):
        MathRecord.from_dict({**payload, "typo": 1})
    with pytest.raises(ValueError, match="missing record fields"):
        MathRecord.from_dict({key: value for key, value in payload.items() if key != "answer"})
    with pytest.raises(ValueError, match="must be strings"):
        MathRecord.from_dict({**payload, "answer": 3})
    with pytest.raises(ValueError, match="difficulty must be an integer"):
        MathRecord.from_dict({**payload, "difficulty": "3"})


def test_generation_is_byte_level_deterministic():
    first = [record.to_dict() for record in generate_synthetic_records(split="train", count=30, seed="s")]
    second = [record.to_dict() for record in generate_synthetic_records(split="train", count=30, seed="s")]
    different = [record.to_dict() for record in generate_synthetic_records(split="train", count=30, seed="t")]
    assert first == second
    assert first != different
    assert len({item["family"] for item in first}) == 6


def test_ood_templates_are_disjoint_and_enforced():
    train = generate_synthetic_records(split="train", count=24, seed="train")
    ood = generate_synthetic_records(split="ood", count=24, seed="eval")
    assert {record.template_id for record in train} <= synthetic_template_ids("in_domain")
    assert {record.template_id for record in ood} <= synthetic_template_ids("ood")
    assert synthetic_template_ids("in_domain").isdisjoint(synthetic_template_ids("ood"))
    result = validate_split_isolation({"train": train, "ood": ood})
    assert result["records"] == 48

    leaked_payload = ood[0].to_dict()
    leaked_payload["split"] = "train"
    leaked = MathRecord.from_dict(leaked_payload)
    with pytest.raises(ValueError, match="held-out OOD template"):
        validate_split_isolation({"train": [leaked]})


def test_dataset_manifest_detects_tampering(tmp_path):
    output = tmp_path / "dataset"
    manifest = generate_synthetic_dataset(
        output,
        sizes={"train": 24, "dev": 12, "test": 12, "ood": 12},
        train_seed="train",
        eval_seed="private-eval",
    )
    assert manifest["seed_fingerprints"]["eval"] != "private-eval"
    assert manifest["evaluation_seed_status"] == "private_user_supplied"
    assert validate_dataset_directory(output)["isolation"]["records"] == 60

    with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_dataset_directory(output)


def test_full_poc_sizes_resample_cross_split_collisions(tmp_path):
    """Regression for a collision observed at the planned 800/200/400/400 scale."""

    output = tmp_path / "full"
    manifest = generate_synthetic_dataset(
        output,
        sizes={"ood": 400, "test": 400, "dev": 200, "train": 800},
        train_seed="1729-public-train",
        eval_seed="PUBLIC-SMOKE-EVAL-SEED-NOT-FOR-RESULTS",
    )
    # Generation order is canonical even though the input mapping was reversed.
    assert list(manifest["files"]) == ["train", "dev", "test", "ood"]
    assert manifest["evaluation_seed_status"] == "public_smoke_only"
    validated = validate_dataset_directory(output)
    assert validated["isolation"]["records"] == 1800
    assert validated["isolation"]["unique_semantics"] == 1800
    repeated = generate_synthetic_dataset(
        tmp_path / "full-repeat",
        sizes={"train": 800, "dev": 200, "test": 400, "ood": 400},
        train_seed="1729-public-train",
        eval_seed="PUBLIC-SMOKE-EVAL-SEED-NOT-FOR-RESULTS",
    )
    assert repeated == manifest


def test_jsonl_error_reports_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"not": "the schema"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl:1"):
        load_jsonl(path)


def test_big_math_requires_allowlist_and_blocks_eval_families():
    rows = [
        {"problem": "What is 1 + 2?", "answer": "3", "source": "Orca-Math", "domain": "arithmetic"},
        {
            "prompt": "What is 3 + 4?",
            "solution": "7",
            "source": "big_math",
            "domain": ["Mathematics", "Arithmetic"],
            "llama8b_solve_rate": 0.5,
        },
        {"problem": "What is 2 + 2?", "answer": "4", "source": "MATH", "domain": "algebra"},
        {"problem": "Symbolic", "answer": "x^2", "source": "Orca-Math", "domain": "algebra"},
    ]
    records, stats = convert_big_math_rows(
        rows, allowed_sources=["Orca-Math"], limit=10, seed="external"
    )
    assert len(records) == 1
    assert records[0].split == "external_train"
    assert records[0].source == "big_math/orca-math"
    assert stats["source_filtered"] == 2
    assert stats["non_numeric_answer"] == 1

    with pytest.raises(ValueError, match="explicitly provided"):
        convert_big_math_rows(rows, allowed_sources=[], limit=10, seed="external")
    with pytest.raises(ValueError, match="evaluation-family"):
        convert_big_math_rows(rows, allowed_sources=["MATH"], limit=10, seed="external")

    processed, filtered = convert_big_math_rows(
        rows,
        allowed_sources=["big_math"],
        limit=10,
        seed="external",
        upstream_dataset="open-r1/Big-Math-RL-Verified-Processed",
        min_solve_rate=0.1,
        max_solve_rate=0.8,
    )
    assert len(processed) == 1
    assert processed[0].prompt == "What is 3 + 4?"
    assert processed[0].answer == "7"
    assert processed[0].family == "Arithmetic"
    assert processed[0].metadata["upstream_dataset"] == "open-r1/Big-Math-RL-Verified-Processed"
    assert filtered["accepted"] == 1


def test_big_math_stream_scan_is_bounded_and_reservoir_is_deterministic():
    def bounded_rows():
        for index in range(5):
            yield {
                "prompt": f"What is {index} plus 1?",
                "solution": str(index + 1),
                "source": "big_math",
                "domain": "arithmetic",
                "llama8b_solve_rate": 0.5,
            }
        raise AssertionError("converter requested a row beyond max_rows_scanned")

    first, stats = convert_big_math_rows(
        bounded_rows(),
        allowed_sources=["big_math"],
        limit=2,
        seed="reservoir",
        max_rows_scanned=5,
        require_limit=True,
    )
    second, repeated_stats = convert_big_math_rows(
        bounded_rows(),
        allowed_sources=["big_math"],
        limit=2,
        seed="reservoir",
        max_rows_scanned=5,
        require_limit=True,
    )
    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]
    assert repeated_stats == stats
    assert stats["rows_scanned"] == 5
    assert stats["max_rows_scanned"] == 5
    assert stats["scan_cap_reached"] == 1
    assert stats["eligible_numeric_rows"] == 5
    assert stats["accepted"] == 2

    with pytest.raises(ValueError, match=r"requested 6.*found only 5.*cap 5"):
        convert_big_math_rows(
            bounded_rows(),
            allowed_sources=["big_math"],
            limit=6,
            seed="reservoir",
            max_rows_scanned=5,
            require_limit=True,
        )


def test_checked_in_sample_is_valid():
    project_root = Path(__file__).resolve().parents[1]
    records = load_jsonl(project_root / "data" / "sample" / "sample.jsonl")
    assert len(records) == 6
    assert {record.family for record in records} == {
        "affine_equation",
        "inventory_value",
        "nested_arithmetic",
        "ratio_share",
        "rational_sum",
        "reverse_percent",
    }
