#!/usr/bin/env python3
"""Evaluate particle, sampling, and Q-selection metrics on sealed splits."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

from _common import absolute_from_project, apply_overrides, load_yaml_config, print_json

from hrm_particle.data import validate_dataset_directory  # noqa: E402


def _call_evaluate_from_config(
    config: dict[str, Any],
    *,
    checkpoint: Path | None,
    output_dir: Path,
) -> Any:
    try:
        from hrm_particle.evaluate import evaluate_from_config
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "hrm_particle.evaluate.evaluate_from_config is unavailable. Install the project "
            "with `pip install -e .` and confirm the checkout is complete."
        ) from exc
    parameters = inspect.signature(evaluate_from_config).parameters
    kwargs: dict[str, Any] = {}
    if "checkpoint" in parameters:
        kwargs["checkpoint"] = checkpoint
    elif "checkpoint_path" in parameters:
        kwargs["checkpoint_path"] = checkpoint
    if "output_dir" in parameters:
        kwargs["output_dir"] = output_dir
    return evaluate_from_config(config, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", default="runs/eval")
    parser.add_argument("--split", choices=("dev", "test", "ood"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config, config_path = load_yaml_config(args.config)
        config = apply_overrides(config, args.set)
        if args.split:
            config.setdefault("evaluation", {})["splits"] = [args.split]
        data = config.get("data")
        if not isinstance(data, dict) or not data.get("directory"):
            raise ValueError("config.data.directory is required")
        validated_data = validate_dataset_directory(absolute_from_project(data["directory"]))
        isolation = validated_data["isolation"]
        evaluation_seed_status = str(
            validated_data["manifest"].get("evaluation_seed_status", "unknown")
        )
        checkpoint = absolute_from_project(args.checkpoint) if args.checkpoint else None
        if checkpoint is not None and not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        output = absolute_from_project(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        print_json(
            {
                "config": str(config_path),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "output_dir": str(output),
                "data": isolation,
                "evaluation_seed_status": evaluation_seed_status,
                "evaluation_warning": (
                    "These are public smoke holdouts, not private result evidence."
                    if evaluation_seed_status == "public_smoke_only"
                    else None
                ),
                "splits": config.get("evaluation", {}).get("splits"),
                "dry_run": args.dry_run,
            }
        )
        if args.dry_run:
            return 0
        result = _call_evaluate_from_config(config, checkpoint=checkpoint, output_dir=output)
        print_json({"status": "complete", "result": result})
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
