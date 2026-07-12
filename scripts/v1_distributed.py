#!/usr/bin/env python3
"""Torchrun entry point for the HRM-Text-1B particle V1 notebook."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hrm_particle.v1_distributed import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

