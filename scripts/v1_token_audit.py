#!/usr/bin/env python3
"""Audit every prepared prompt against HRM-Text-1B's context window."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hrm_particle.v1_preflight import audit_prepared_context_lengths  # noqa: E402
from hrm_particle.v1_data import load_evalplus_mbpp_prompts  # noqa: E402


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
    parser.add_argument("--output", default="runs/v1_2gpu_poc/token-audit.json")
    args = parser.parse_args()
    try:
        import yaml
        from transformers import AutoTokenizer

        config_path = Path(args.config).expanduser().resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = config["model"]
        mbpp_prompts, evalplus_version = load_evalplus_mbpp_prompts()
        expected_evalplus = str(config["evaluation"]["evalplus_version"])
        if evalplus_version != expected_evalplus:
            raise RuntimeError(
                f"EvalPlus version mismatch: installed {evalplus_version}, "
                f"expected {expected_evalplus}"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model["pretrained_model_name_or_path"]),
            revision=str(model["revision"]),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
        )
        data_directory = Path(config["paths"]["data_directory"])
        if not data_directory.is_absolute():
            data_directory = PROJECT_ROOT / data_directory
        report = audit_prepared_context_lengths(
            tokenizer=tokenizer,
            config=config,
            data_directory=data_directory,
            mbpp_prompts=mbpp_prompts,
        )
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
