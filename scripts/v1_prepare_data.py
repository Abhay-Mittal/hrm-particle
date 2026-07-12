#!/usr/bin/env python3
"""Prepare pinned, decontaminated V1 math/code prompt pools."""

from __future__ import annotations

import argparse
import json
import sys

from _common import absolute_from_project, print_json

from hrm_particle.v1_data import PRESETS, prepare_v1_dataset, validate_v1_directory  # noqa: E402


def command_prepare(args: argparse.Namespace) -> int:
    output = absolute_from_project(args.output_dir)
    result = prepare_v1_dataset(
        output,
        preset=args.preset,
        seed=args.seed,
        include_mbpp_plus_contamination=args.include_mbpp_plus_contamination,
    )
    print_json(
        {
            "output_dir": result["output_dir"],
            "manifest_sha256": result["manifest_sha256"],
            "files": result["files"],
            "q_warm": result["q_warm"],
            "decontamination": result["decontamination"],
        }
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    print_json(validate_v1_directory(absolute_from_project(args.data_dir)))
    return 0


def command_show_preset(args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    print(
        json.dumps(
            {
                "preset": args.preset,
                "q_warm": dict(preset.q_warm),
                "q_prompt_groups": preset.q_prompt_groups,
                "q_candidates_at_k4": preset.q_prompt_groups * 4,
                "rl_train": dict(preset.rl_train),
                "eval": dict(preset.eval),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="download, pin, filter, and write V1 data")
    prepare.add_argument("--output-dir", default="data/v1")
    prepare.add_argument("--preset", choices=sorted(PRESETS), default="full")
    prepare.add_argument("--seed", default="1729-v1")
    prepare.add_argument(
        "--include-mbpp-plus-contamination",
        action="store_true",
        help="also exclude 13-token overlaps with EvalPlus MBPP+ prompts",
    )
    prepare.set_defaults(func=command_prepare)

    validate = commands.add_parser("validate", help="verify schemas and SHA256 checksums")
    validate.add_argument("--data-dir", default="data/v1")
    validate.set_defaults(func=command_validate)

    show = commands.add_parser("show-preset", help="print planned counts without network access")
    show.add_argument("--preset", choices=sorted(PRESETS), default="full")
    show.set_defaults(func=command_show_preset)
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
