"""Shared pytest configuration.

Tests are deterministic and offline unless explicitly marked ``network`` or
``gpu``.  The default suite never downloads HRM-Text weights.
"""

from __future__ import annotations

import random

import pytest


@pytest.fixture(autouse=True)
def deterministic_seed():
    random.seed(1234)
    try:
        import torch

        torch.manual_seed(1234)
    except ImportError:
        pass

