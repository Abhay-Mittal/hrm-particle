#!/usr/bin/env python3
"""Prepare isolated synthetic data or an explicitly filtered Big-Math subset."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path

from _common import absolute_from_project, print_json

from hrm_particle.data import (  # noqa: E402
    DEFAULT_BIGMATH_EVAL_SOURCE_DENYLIST,
    PUBLIC_SMOKE_EVAL_SEED,
    convert_big_math_rows,
    generate_synthetic_dataset,
    validate_dataset_directory,
    write_jsonl,
)


def _private_eval_seed(args: argparse.Namespace) -> str:
    supplied = [args.eval_seed is not None, args.eval_seed_file is not None]
    if sum(supplied) > 1:
        raise ValueError("provide only one of --eval-seed and --eval-seed-file")
    if args.eval_seed is not None:
        return str(args.eval_seed)
    if args.eval_seed_file is not None:
        path = Path(args.eval_seed_file).expanduser().resolve()
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"evaluation seed file does not exist: {path}") from exc
        if mode & 0o077:
            raise PermissionError(
                f"evaluation seed file must not be group/world accessible (run chmod 600): {path}"
            )
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"evaluation seed file is empty: {path}")
        return value
    environment = os.getenv("HRM_EVAL_SEED")
    if environment:
        return environment
    if args.public_eval_seed_for_smoke:
        return PUBLIC_SMOKE_EVAL_SEED
    raise ValueError(
        "a private evaluation seed is required: use --eval-seed-file or HRM_EVAL_SEED; "
        "--public-eval-seed-for-smoke is only for plumbing tests"
    )


def command_synthetic(args: argparse.Namespace) -> int:
    eval_seed = _private_eval_seed(args)
    output = absolute_from_project(args.output_dir)
    sizes = {
        "train": args.train_count,
        "dev": args.dev_count,
        "test": args.test_count,
        "ood": args.ood_count,
    }
    manifest = generate_synthetic_dataset(
        output,
        sizes=sizes,
        train_seed=args.train_seed,
        eval_seed=eval_seed,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
    )
    status = str(manifest["evaluation_seed_status"])
    if status == "public_smoke_only":
        status_text = (
            "PUBLIC SMOKE DATA ONLY\n\n"
            "The bundled dev/test/OOD rows use a documented public seed. They are ready "
            "for plumbing and preliminary training, but are not private paper evidence.\n"
        )
    else:
        status_text = (
            "PRIVATE USER-SUPPLIED EVALUATION SEED\n\n"
            "The manifest contains only a one-way seed fingerprint. Keep the raw seed "
            "private and preserve it separately if exact regeneration matters.\n"
        )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output, delete=False, prefix=".data-status."
    ) as handle:
        temporary_status = Path(handle.name)
        handle.write(status_text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_status, output / "DATA_STATUS.txt")
    print_json({"output_dir": output, "manifest": manifest})
    return 0


def command_validate(args: argparse.Namespace) -> int:
    result = validate_dataset_directory(absolute_from_project(args.data_dir))
    print_json(result)
    return 0


def command_make_seed(args: argparse.Namespace) -> int:
    path = absolute_from_project(args.output)
    if path.exists() and not args.force:
        raise FileExistsError(f"refusing to replace existing secret: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (0 if args.force else os.O_EXCL)
    if args.force:
        flags |= os.O_TRUNC
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, (secrets.token_hex(32) + "\n").encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)
    print(f"Wrote private evaluation seed to {path} (mode 0600). Do not commit it.")
    return 0


def command_bigmath(args: argparse.Namespace) -> int:
    output = absolute_from_project(args.output)
    if output.name in {"train.jsonl", "dev.jsonl", "test.jsonl", "ood.jsonl"}:
        raise ValueError(
            "Big-Math must be written to a separately named external-training file; "
            "it is never appended to synthetic train/evaluation splits"
        )
    if not args.source:
        raise ValueError("repeat --source for every explicitly approved Big-Math source")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the optional 'datasets' package is required; install requirements-data.txt"
        ) from exc

    token = os.getenv("HF_TOKEN")
    rows = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
        token=token,
    )
    records, stats = convert_big_math_rows(
        rows,
        allowed_sources=args.source,
        limit=args.limit,
        seed=args.seed,
        upstream_dataset=args.dataset,
        min_solve_rate=args.min_solve_rate,
        max_solve_rate=args.max_solve_rate,
        max_rows_scanned=args.max_rows_scanned,
        require_limit=not args.allow_fewer,
    )
    if not records:
        raise RuntimeError(
            "no strict numeric rows survived filtering; check source spellings and gated-dataset access"
        )
    result = write_jsonl(output, records)
    manifest = {
        "kind": "external_training_only",
        "upstream_dataset": args.dataset,
        "upstream_config": args.dataset_config,
        "upstream_split": args.dataset_split,
        "allowed_sources": sorted(args.source),
        "frontier_solve_rate": {"min": args.min_solve_rate, "max": args.max_solve_rate},
        "scan": {
            "rows_scanned": stats["rows_scanned"],
            "max_rows_scanned": stats["max_rows_scanned"],
            "scan_cap_reached": bool(stats["scan_cap_reached"]),
        },
        "blocked_eval_sources": sorted(DEFAULT_BIGMATH_EVAL_SOURCE_DENYLIST),
        "seed_fingerprint": __import__("hashlib").sha256(str(args.seed).encode()).hexdigest()[:16],
        "file": {"path": output.name, **result},
        "filter_stats": stats,
        "warning": "This file is not part of synthetic test/OOD data and must never be merged into them.",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest_path.parent,
        delete=False,
        prefix=f".{manifest_path.name}.",
    ) as handle:
        temporary_manifest = Path(handle.name)
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_manifest, manifest_path)
    print_json({"output": output, "manifest": manifest_path, **manifest})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser("synthetic", help="generate exact arithmetic JSONL splits")
    synthetic.add_argument("--output-dir", default="data/processed")
    synthetic.add_argument("--train-count", type=int, default=800)
    synthetic.add_argument("--dev-count", type=int, default=200)
    synthetic.add_argument("--test-count", type=int, default=400)
    synthetic.add_argument("--ood-count", type=int, default=400)
    synthetic.add_argument("--train-seed", default="1729-public-train")
    synthetic.add_argument("--eval-seed")
    synthetic.add_argument("--eval-seed-file")
    synthetic.add_argument("--public-eval-seed-for-smoke", action="store_true")
    synthetic.add_argument("--min-difficulty", type=int, default=2)
    synthetic.add_argument("--max-difficulty", type=int, default=5)
    synthetic.set_defaults(func=command_synthetic)

    validate = commands.add_parser("validate", help="verify checksums, schema, and split isolation")
    validate.add_argument("--data-dir", default="data/processed")
    validate.set_defaults(func=command_validate)

    make_seed = commands.add_parser("make-eval-seed", help="create a private reproducible eval seed")
    make_seed.add_argument("--output", default="secrets/eval_seed.txt")
    make_seed.add_argument("--force", action="store_true")
    make_seed.set_defaults(func=command_make_seed)

    bigmath = commands.add_parser("bigmath", help="download a separate strict-numeric training subset")
    bigmath.add_argument("--dataset", default="open-r1/Big-Math-RL-Verified-Processed")
    bigmath.add_argument(
        "--dataset-config",
        default="all",
        help="Hugging Face dataset config/subset (Open-R1's complete split is 'all')",
    )
    bigmath.add_argument("--dataset-split", default="train")
    bigmath.add_argument("--source", action="append", default=[])
    bigmath.add_argument("--limit", type=int, default=800)
    bigmath.add_argument("--max-rows-scanned", type=int, default=100_000)
    bigmath.add_argument(
        "--allow-fewer",
        action="store_true",
        help="write fewer than --limit only when strict filters/cap yield too few rows",
    )
    bigmath.add_argument("--min-solve-rate", type=float, default=0.1)
    bigmath.add_argument("--max-solve-rate", type=float, default=0.8)
    bigmath.add_argument("--seed", default="1729-bigmath")
    bigmath.add_argument("--output", default="data/external/big_math_train.jsonl")
    bigmath.set_defaults(func=command_bigmath)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
