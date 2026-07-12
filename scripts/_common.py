"""Shared command-line helpers (kept dependency-light for Runpod)."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def absolute_from_project(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def load_yaml_config(path: str | os.PathLike[str]) -> tuple[dict[str, Any], Path]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("PyYAML is required; run: pip install -r requirements.txt") from exc

    resolved = absolute_from_project(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"config does not exist: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {resolved}")
    return payload, resolved


def deep_set(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current: dict[str, Any] = config
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override {dotted_key}: {part} is not a mapping")
        current = child
    current[parts[-1]] = value


def parse_scalar(value: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(value)
    except ImportError:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value


def apply_overrides(config: Mapping[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        if not key.strip():
            raise ValueError("override key cannot be empty")
        deep_set(updated, key.strip(), parse_scalar(raw))
    return updated


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def validate_common_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate invariants that should hold before loading a GPU model."""

    model = _mapping(config, "model")
    adapter = _mapping(config, "adapter")
    particles = _mapping(config, "particles")
    generation = _mapping(config, "generation")
    data = _mapping(config, "data")
    q_head = _mapping(config, "q_head")
    budget = _mapping(config, "budget")

    if not model.get("pretrained_model_name_or_path"):
        raise ValueError("model.pretrained_model_name_or_path is required")
    if model.get("freeze_backbone") is not True:
        raise ValueError("POC-A requires model.freeze_backbone=true")
    if model.get("gradient_checkpointing", False):
        raise ValueError("HRM particle hooks do not support model.gradient_checkpointing=true")
    if adapter.get("zero_latent_is_exact_zero") is not True:
        raise ValueError("adapter.zero_latent_is_exact_zero must remain true")
    if int(particles.get("count", 0)) != 4:
        raise ValueError("POC v1 requires particles.count=4")
    if int(particles.get("anchor_index", -1)) != 0:
        raise ValueError("POC configs require particles.anchor_index=0")
    if particles.get("reuse_latent_across_response") is not True:
        raise ValueError("one latent must be reused across the whole response")
    if q_head.get("detach_actor_states") is not True:
        raise ValueError("q_head.detach_actor_states must be true")
    if q_head.get("use_q_as_actor_reward", False):
        raise ValueError("Q must not be used as the actor reward")
    if int(generation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if int(generation.get("max_new_tokens", 0)) > 128:
        raise ValueError("POC budget guard: generation.max_new_tokens cannot exceed 128")
    if float(generation.get("top_p", 0.0)) != 1.0:
        raise ValueError(
            "POC training requires generation.top_p=1.0 so particle actions have finite "
            "support under the zero-latent KL reference"
        )
    if not data.get("directory"):
        raise ValueError("data.directory is required")

    hourly = float(budget.get("hourly_usd", 0.0))
    hours = float(budget.get("max_gpu_hours", 0.0))
    maximum = float(budget.get("max_cost_usd", 0.0))
    if hourly <= 0 or hours <= 0 or maximum <= 0:
        raise ValueError("budget hourly_usd, max_gpu_hours, and max_cost_usd must be positive")
    projected = hourly * hours
    if projected > maximum + 1e-9:
        raise ValueError(
            f"budget is inconsistent: {hours:g}h * ${hourly:g}/h = ${projected:.2f}, "
            f"above max_cost_usd=${maximum:.2f}"
        )
    stop_buffer_minutes = float(budget.get("stop_buffer_minutes", 0.0))
    if stop_buffer_minutes < 0 or stop_buffer_minutes >= hours * 60:
        raise ValueError("budget.stop_buffer_minutes must be non-negative and shorter than the run")
    return {
        "particle_count": int(particles["count"]),
        "max_new_tokens": int(generation["max_new_tokens"]),
        "projected_cost_usd": projected,
        "max_cost_usd": maximum,
        "max_gpu_hours": hours,
        "data_directory": str(data["directory"]),
    }


def apply_budget_ceiling(
    config: Mapping[str, Any],
    *,
    max_cost_usd: float | None,
    max_gpu_hours: float | None,
) -> dict[str, Any]:
    """Allow CLI overrides only when they tighten the checked-in ceiling."""

    updated = copy.deepcopy(dict(config))
    budget = updated.setdefault("budget", {})
    if not isinstance(budget, dict):
        raise ValueError("config.budget must be a mapping")
    configured_cost = float(budget.get("max_cost_usd", 0.0))
    configured_hours = float(budget.get("max_gpu_hours", 0.0))
    if max_cost_usd is not None:
        if max_cost_usd <= 0 or max_cost_usd > configured_cost:
            raise ValueError(
                f"--max-cost-usd may only lower the config ceiling (${configured_cost:g})"
            )
        budget["max_cost_usd"] = max_cost_usd
    if max_gpu_hours is not None:
        if max_gpu_hours <= 0 or max_gpu_hours > configured_hours:
            raise ValueError(
                f"--max-gpu-hours may only lower the config ceiling ({configured_hours:g}h)"
            )
        budget["max_gpu_hours"] = max_gpu_hours
    # Tightening cost can require tightening hours to maintain the invariant.
    hourly = float(budget["hourly_usd"])
    budget["max_gpu_hours"] = min(float(budget["max_gpu_hours"]), float(budget["max_cost_usd"]) / hourly)
    return updated


def set_runtime_budget_environment(config: Mapping[str, Any]) -> None:
    budget = _mapping(config, "budget")
    buffer_seconds = float(budget.get("stop_buffer_minutes", 0.0)) * 60.0
    seconds = float(budget["max_gpu_hours"]) * 3600 - buffer_seconds
    if seconds <= 0:
        raise ValueError("budget stop buffer must be shorter than max_gpu_hours")
    deadline = time.time() + seconds
    os.environ["HRM_PARTICLE_DEADLINE_UNIX"] = str(deadline)
    os.environ["HRM_PARTICLE_MAX_COST_USD"] = str(float(budget["max_cost_usd"]))
    os.environ["HRM_PARTICLE_GPU_HOURLY_USD"] = str(float(budget["hourly_usd"]))


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
