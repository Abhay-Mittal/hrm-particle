#!/usr/bin/env python3
"""Summarize the four pinned EvalPlus MBPP+ result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hrm_particle.v1_evalplus import summarize_evalplus_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        default=str(PROJECT_ROOT / "runs" / "v1_2gpu_poc" / "evalplus_mbpp"),
    )
    args = parser.parse_args()
    try:
        summary = summarize_evalplus_directory(Path(args.directory))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
