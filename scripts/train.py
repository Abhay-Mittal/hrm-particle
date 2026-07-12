#!/usr/bin/env python3
"""Budget-guarded launcher for adapter-only particle training."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

from _common import (
    absolute_from_project,
    apply_budget_ceiling,
    apply_overrides,
    load_yaml_config,
    print_json,
    set_runtime_budget_environment,
    validate_common_config,
)

from hrm_particle.data import validate_dataset_directory  # noqa: E402


def _call_train_from_config(
    config: dict[str, Any],
    *,
    output_dir: Path,
    resume_from: str | None,
) -> Any:
    try:
        from hrm_particle.trainer import train_from_config
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "hrm_particle.trainer.train_from_config is unavailable. Install the project with "
            "`pip install -e '.[train]'` and confirm the checkout is complete."
        ) from exc
    parameters = inspect.signature(train_from_config).parameters
    kwargs: dict[str, Any] = {}
    if "output_dir" in parameters:
        kwargs["output_dir"] = output_dir
    if "resume_from" in parameters:
        kwargs["resume_from"] = resume_from
    elif "resume_from_checkpoint" in parameters:
        kwargs["resume_from_checkpoint"] = resume_from
    return train_from_config(config, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/poc_h100.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--max-gpu-hours", type=float)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="validate without loading a model")
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="not recommended; skip manifest/checksum/isolation validation",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config, config_path = load_yaml_config(args.config)
        budget_overrides = [
            item for item in args.set if item.split("=", 1)[0].strip().startswith("budget.")
        ]
        if budget_overrides:
            raise ValueError(
                "budget.* cannot be changed through --set; edit and review the YAML, then use "
                "--max-cost-usd/--max-gpu-hours only to tighten its ceiling"
            )
        config = apply_overrides(config, args.set)
        config = apply_budget_ceiling(
            config,
            max_cost_usd=args.max_cost_usd,
            max_gpu_hours=args.max_gpu_hours,
        )
        summary = validate_common_config(config)
        data_directory = absolute_from_project(summary["data_directory"])
        data_result = None
        evaluation_seed_status = "not_checked"
        if not args.skip_data_validation:
            validated_data = validate_dataset_directory(data_directory)
            data_result = validated_data["isolation"]
            evaluation_seed_status = str(
                validated_data["manifest"].get("evaluation_seed_status", "unknown")
            )
        configured_output = config.get("output", {}).get("directory", "runs/poc")
        output_dir = absolute_from_project(args.output_dir or configured_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        launch = {
            "config": str(config_path),
            "output_dir": str(output_dir),
            "data": data_result,
            "evaluation_seed_status": evaluation_seed_status,
            "evaluation_warning": (
                "Bundled dev/test/OOD use a public smoke seed; regenerate with a private "
                "evaluation seed before reporting results."
                if evaluation_seed_status == "public_smoke_only"
                else None
            ),
            "budget": {
                "projected_cost_usd": summary["projected_cost_usd"],
                "max_cost_usd": summary["max_cost_usd"],
                "max_gpu_hours": summary["max_gpu_hours"],
                "note": "Also set a provider-side Runpod spend/timeout limit.",
            },
            "dry_run": args.dry_run,
        }
        print_json(launch)
        if args.dry_run:
            return 0
        set_runtime_budget_environment(config)
        result = _call_train_from_config(
            config,
            output_dir=output_dir,
            resume_from=args.resume_from,
        )
        print_json({"status": "complete", "result": result})
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
