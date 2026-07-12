"""Small, resumable checkpoints that intentionally exclude the frozen 1B base."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn


CHECKPOINT_FORMAT = "hrm-particle-poc"
CHECKPOINT_VERSION = 1


def _config_dict(config: Any) -> Any:
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dataclass, mapping, or None")


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: nn.Module,
    q_head: nn.Module,
    step: int,
    actor_optimizer: Optional[torch.optim.Optimizer] = None,
    q_optimizer: Optional[torch.optim.Optimizer] = None,
    rollout_engine: Any = None,
    config: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically save trainable modules and resume state, never base weights."""

    if step < 0:
        raise ValueError("step must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "step": int(step),
        "adapter": adapter.state_dict(),
        "q_head": q_head.state_dict(),
        "actor_optimizer": (
            actor_optimizer.state_dict() if actor_optimizer is not None else None
        ),
        "q_optimizer": q_optimizer.state_dict() if q_optimizer is not None else None,
        "config": _config_dict(config),
        "metadata": dict(metadata or {}),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "rollout_rng_state": (
            rollout_engine.rng_state_dict() if rollout_engine is not None else None
        ),
    }
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: nn.Module,
    q_head: nn.Module,
    actor_optimizer: Optional[torch.optim.Optimizer] = None,
    q_optimizer: Optional[torch.optim.Optimizer] = None,
    rollout_engine: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a POC checkpoint and return its step/config/metadata."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not an HRM particle POC checkpoint")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {payload.get('version')!r}")

    adapter.load_state_dict(payload["adapter"], strict=strict)
    q_head.load_state_dict(payload["q_head"], strict=strict)
    if actor_optimizer is not None:
        state = payload.get("actor_optimizer")
        if state is None:
            raise ValueError("checkpoint contains no actor optimizer state")
        actor_optimizer.load_state_dict(state)
    if q_optimizer is not None:
        state = payload.get("q_optimizer")
        if state is None:
            raise ValueError("checkpoint contains no Q optimizer state")
        q_optimizer.load_state_dict(state)

    if restore_rng:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_state = payload.get("cuda_rng_state_all")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
        rollout_state = payload.get("rollout_rng_state")
        if rollout_engine is not None and rollout_state is not None:
            rollout_engine.load_rng_state_dict(rollout_state)
    return {
        "step": int(payload["step"]),
        "config": payload.get("config"),
        "metadata": payload.get("metadata", {}),
    }


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "load_checkpoint",
    "save_checkpoint",
]
