#!/usr/bin/env python3
"""Verify the pinned Open-R1 install and strict remote sandbox semantics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hrm_particle.v1_dependencies import (  # noqa: E402
    package_versions,
    require_installed_vcs_commit,
)
from hrm_particle.v1_rewards import (  # noqa: E402
    STRICT_CODE_HARNESS_VERSION,
    OpenR1SandboxCodeScorer,
    run_remote_code_sandbox_canary,
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1_2gpu_poc.yaml")
    parser.add_argument("--output", default="runs/v1_2gpu_poc/code-provider-canary.json")
    parser.add_argument("--skip-resource-checks", action="store_true")
    args = parser.parse_args()
    try:
        import yaml

        config_path = Path(args.config).expanduser().resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        verification = config["verification"]
        expected_commit = str(verification["open_r1_commit"])
        observed_commit = require_installed_vcs_commit("open-r1", expected_commit)
        scorer = OpenR1SandboxCodeScorer(
            provider=str(verification["code_provider"]),
            num_parallel=int(verification["code_parallelism_per_rank"]),
            binary_reward_weight=float(config["rl"]["code_binary_reward_weight"]),
        )
        checks = run_remote_code_sandbox_canary(
            scorer,
            include_resource_checks=not args.skip_resource_checks,
        )
        report = {
            "format": "hrm-particle-v1-code-provider-canary",
            "checked_at_unix": time.time(),
            "provider": scorer.provider,
            "harness_version": STRICT_CODE_HARNESS_VERSION,
            "open_r1_commit": observed_commit,
            "package_versions": package_versions(),
            "checks": checks,
        }
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        _atomic_json(output.resolve(), report)
    except (FileNotFoundError, ImportError, KeyError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
