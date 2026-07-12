"""Pinned dependency provenance and remote verifier preflight helpers."""

from __future__ import annotations

import importlib.metadata
import json
import re
from typing import Any


PINNED_PACKAGE_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "safetensors",
    "math-verify",
    "open-r1",
    "evalplus",
)


def package_versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for name in PINNED_PACKAGE_NAMES:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "missing"
    return values


def installed_vcs_commit(distribution_name: str) -> str:
    """Read the immutable PEP 610 VCS commit recorded by a pip Git install."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is not installed: {distribution_name}") from exc
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise RuntimeError(
            f"{distribution_name} has no direct_url.json; install it from the pinned Git commit"
        )
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{distribution_name} has malformed PEP 610 provenance") from exc
    commit = (
        payload.get("vcs_info", {}).get("commit_id")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError(f"{distribution_name} does not record a full Git commit")
    return commit


def require_installed_vcs_commit(distribution_name: str, expected: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", str(expected)) is None:
        raise ValueError("expected VCS revision must be a full lowercase Git SHA")
    observed = installed_vcs_commit(distribution_name)
    if observed != expected:
        raise RuntimeError(
            f"{distribution_name} commit mismatch: installed {observed}, expected {expected}"
        )
    return observed


__all__ = [
    "installed_vcs_commit",
    "package_versions",
    "require_installed_vcs_commit",
]
