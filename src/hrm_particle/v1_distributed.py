"""Restartable multi-rank runner for the HRM-Text-1B particle V1.

The notebook deliberately launches this module through ``torchrun`` instead of
initializing CUDA in the notebook kernel.  One BF16 model replica lives on each
GPU.  Each rank expands its own prompt to K=4 branches and DDP synchronizes only
the small particle adapter and correctness Q head.

Generated Python is never executed here.  Training code rewards are delegated
to the explicitly configured Open-R1 E2B/Morph provider, and MBPP+ samples are
written for the official EvalPlus Docker image.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .adapter import ParticleAdapter, ParticleAdapterConfig, SharedQHead
from .evaluate import paired_bootstrap_delta
from .gaussian import GaussianParticleAdapter, GaussianParticleConfig
from .model import ParticleHrmForCausalLM
from .objectives import (
    anchor_rescue_advantages,
    clipped_token_policy_loss,
    supervised_q_loss,
)
from .rollout import (
    ParticleRollout,
    ParticleRolloutEngine,
    RolloutConfig,
    RolloutExample,
    normalized_gaussian_latents,
    score_actions,
)
from .v1_data import load_evalplus_mbpp_prompts, validate_v1_directory
from .v1_dependencies import package_versions, require_installed_vcs_commit
from .v1_memory import GIB, estimate_batch_from_peak, valid_batch_candidates
from .v1_rewards import (
    STRICT_CODE_HARNESS_VERSION,
    MathVerifyScorer,
    OpenR1SandboxCodeScorer,
    RewardRouter,
    rescore_particle_rollout,
)
from .v1_utils import (
    BF16MasterAdamW,
    Q_SPLIT_ALGORITHM,
    atomic_torch_save,
    calibrate_q_margin,
    code_signal_gate,
    extract_python_code,
    select_with_anchor_fallback,
    sha256_file,
    stratified_prompt_split_masks,
    stratified_prompt_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODEL_PARAMETERS = 1_182_795_264
EXPECTED_Q_PARAMETERS = 786_945
CHECKPOINT_FORMAT = "hrm-particle-v1"
CHECKPOINT_VERSION = 2
INITIAL_ADAPTER_FORMAT = "hrm-particle-v1-initial-adapter"
INITIAL_ADAPTER_VERSION = 1
MEMORY_PLAN_FORMAT = "hrm-particle-v1-memory-plan"
MEMORY_PLAN_VERSION = 1
NOISE_SCALE_FORMAT = "hrm-particle-v1-noise-scale-selection"
NOISE_SCALE_VERSION = 1


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required; install requirements-v1.txt") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return payload


def _project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _particle_mode(config: Mapping[str, Any]) -> str:
    mode = str(config["particles"].get("mode", "learned")).lower()
    if mode not in {"gaussian", "learned"}:
        raise ValueError("particles mode must be 'gaussian' or 'learned'")
    return mode


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        handle.flush()


def _reconcile_metrics_log(path: Path, checkpoint_step: int) -> float:
    """Drop post-checkpoint/duplicate rows and return cumulative elapsed time."""

    if checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative")
    if not path.exists():
        return 0.0
    by_step: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed metrics row {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"metrics row {path}:{line_number} is not an object")
            raw_step = payload.get("step")
            if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
                raise RuntimeError(f"metrics row {path}:{line_number} has no numeric step")
            numeric_step = float(raw_step)
            if not math.isfinite(numeric_step) or not numeric_step.is_integer():
                raise RuntimeError(f"metrics row {path}:{line_number} has an invalid step")
            step = int(numeric_step)
            if step <= 0:
                raise RuntimeError(f"metrics row {path}:{line_number} has a non-positive step")
            if step <= checkpoint_step:
                # The last complete row wins if an older runner appended a
                # duplicate before this idempotent reconciliation existed.
                by_step[step] = payload

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for step in sorted(by_step):
            handle.write(json.dumps(by_step[step], sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)
    if not by_step:
        return 0.0
    last = by_step[max(by_step)]
    elapsed = float(last.get("elapsed_seconds", 0.0))
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("metrics elapsed_seconds must be finite and non-negative")
    return elapsed


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _configure_runtime(config: Mapping[str, Any]) -> None:
    """Apply runtime controls that would otherwise be inert YAML documentation."""

    runtime = config["runtime"]
    deterministic = runtime["deterministic_algorithms"]
    tf32 = runtime["tf32"]
    if not isinstance(deterministic, bool) or not isinstance(tf32, bool):
        raise ValueError("runtime deterministic_algorithms and tf32 must be booleans")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def _distributed_context(config: Mapping[str, Any], *, require_distributed: bool) -> DistributedContext:
    _configure_runtime(config)
    if not torch.cuda.is_available():
        raise RuntimeError("this stage requires CUDA; use the offline pytest cell first")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "this V1 requires a CUDA GPU with native BF16 support; no GPU model name "
            "is hard-coded"
        )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected = int(config["runtime"]["world_size"])
    if require_distributed and world_size != expected:
        raise RuntimeError(
            f"stage requires torchrun world_size={expected}, observed {world_size}; "
            "do not run it directly in the notebook kernel"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        timeout = int(config["runtime"].get("nccl_timeout_minutes", 30))
        from datetime import timedelta

        dist.init_process_group("nccl", timeout=timedelta(minutes=timeout))
    return DistributedContext(rank, local_rank, world_size, device)


def _barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def _memory_plan_path(config: Mapping[str, Any]) -> Path:
    return _project_path(config["paths"]["memory_plan"])


def _memory_plan_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    token_audit = _project_path(config["paths"]["token_audit"])
    if not token_audit.is_file():
        raise FileNotFoundError(
            f"token audit is missing: {token_audit}; run scripts/v1_token_audit.py first"
        )
    return {
        "config_hash": _canonical_hash(config),
        "model_revision": str(config["model"]["revision"]),
        "token_audit_sha256": sha256_file(token_audit),
        "world_size": int(config["runtime"]["world_size"]),
    }


def _load_memory_plan(
    config: Mapping[str, Any], context: DistributedContext | None = None
) -> dict[str, Any]:
    path = _memory_plan_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"VRAM plan is missing: {path}; run the distributed vram_plan stage first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("VRAM plan root is not an object")
    if (
        payload.get("format") != MEMORY_PLAN_FORMAT
        or payload.get("version") != MEMORY_PLAN_VERSION
    ):
        raise RuntimeError("unsupported VRAM plan format")
    if payload.get("identity") != _memory_plan_identity(config):
        raise RuntimeError(
            "VRAM plan does not match this config/model/data; rerun vram_plan"
        )
    selected = int(payload.get("rl_prompt_micro_batch_size", 0))
    accumulation = int(payload.get("rl_gradient_accumulation_steps", 0))
    rollout_batch = int(payload.get("rollout_prompt_batch_size", 0))
    q_batch = int(payload.get("q_batch_prompt_groups", 0))
    target_per_rank = int(config["memory"]["target_prompt_groups_per_rank_update"])
    if min(selected, accumulation, rollout_batch, q_batch) <= 0:
        raise RuntimeError("VRAM plan contains a non-positive batch value")
    if selected * accumulation != target_per_rank:
        raise RuntimeError("VRAM plan does not preserve the configured effective batch")
    candidates = valid_batch_candidates(
        config["memory"]["candidate_prompt_batch_sizes"],
        target_prompts_per_rank_update=target_per_rank,
    )
    if selected not in candidates:
        raise RuntimeError("VRAM plan selected a batch outside the configured candidates")
    if selected != rollout_batch:
        raise RuntimeError("V1 uses one empirically validated rollout batch across stages")
    if q_batch > int(config["memory"]["maximum_q_batch_prompt_groups"]):
        raise RuntimeError("VRAM plan exceeds the configured Q batch cap")
    if float(payload.get("target_fraction", -1.0)) != float(
        config["memory"]["target_fraction"]
    ) or float(payload.get("hard_limit_fraction", -1.0)) != float(
        config["memory"]["hard_limit_fraction"]
    ):
        raise RuntimeError("VRAM plan target fractions do not match the config")
    if float(payload.get("measured_peak_fraction_max", 1.0)) > float(
        config["memory"]["hard_limit_fraction"]
    ):
        raise RuntimeError("VRAM plan exceeds the configured measured hard cap")
    if context is not None:
        current_total = int(torch.cuda.get_device_properties(context.device).total_memory)
        current_free, _ = torch.cuda.mem_get_info(context.device)
        current_usable = int(current_free) + int(
            torch.cuda.memory_reserved(context.device)
        )
        minimum_planned = int(payload.get("minimum_total_bytes", 0))
        minimum_planned_usable = int(
            payload.get("minimum_usable_capacity_bytes", 0)
        )
        if minimum_planned <= 0:
            raise RuntimeError("VRAM plan does not record measured device capacity")
        if minimum_planned_usable <= 0:
            raise RuntimeError("VRAM plan does not record measured usable capacity")
        reports = payload.get("rank_reports")
        if context.world_size == int(config["runtime"]["world_size"]):
            if not isinstance(reports, list) or context.rank >= len(reports):
                raise RuntimeError("VRAM plan has no capacity record for this rank")
            planned_total = int(reports[context.rank]["physical_total_bytes"])
            if current_total != planned_total:
                raise RuntimeError(
                    "GPU capacity changed since the saved plan; rerun vram_plan with "
                    "--replan-memory before using this accelerator"
                )
            planned_usable = int(reports[context.rank]["usable_capacity_bytes"])
            if current_usable + GIB < planned_usable:
                raise RuntimeError(
                    "current free VRAM is more than 1 GiB below the saved plan; clear GPU "
                    "contention or rerun vram_plan with --replan-memory"
                )
        elif current_total < minimum_planned:
            raise RuntimeError(
                "current GPU has less VRAM than the saved plan; rerun vram_plan with "
                "--replan-memory on this host"
            )
        elif current_usable + GIB < minimum_planned_usable:
            raise RuntimeError(
                "current free VRAM is below the saved plan; clear GPU contention before "
                "continuing"
            )
    return payload


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _seed_everything(seed: int, context: DistributedContext) -> None:
    local_seed = int(seed) + context.rank * 100_003
    random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"prepared data file is missing: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"record {path}:{line_number} is not an object")
            task_type = payload.get("task_type")
            if task_type not in {"math", "code"}:
                raise ValueError(f"record {path}:{line_number} has invalid task_type={task_type!r}")
            if not payload.get("id") or not payload.get("prompt"):
                raise ValueError(f"record {path}:{line_number} is missing id/prompt")
            records.append(payload)
    if not records:
        raise ValueError(f"prepared data file is empty: {path}")
    return records


def _rollout_example(
    record: Mapping[str, Any], prompting: Mapping[str, Any] | None = None
) -> RolloutExample:
    metadata = dict(record.get("metadata") or {})
    metadata["task_type"] = record["task_type"]
    metadata["source"] = record.get("source")
    if record.get("verification_info") is not None:
        metadata["verification_info"] = record["verification_info"]
    suffix = ""
    if prompting is not None:
        suffix_key = "code_suffix" if record["task_type"] == "code" else "math_suffix"
        suffix = str(prompting.get(suffix_key, ""))
    return RolloutExample(
        prompt=str(record["prompt"]).rstrip() + suffix,
        answer=str(
            record.get("answer")
            if record.get("answer") is not None
            else record.get("solution", "")
        ),
        example_id=str(record["id"]),
        metadata=metadata,
    )


def _zero_latents(
    batch_size: int,
    k: int,
    latent_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    **_: Any,
) -> Tensor:
    return torch.zeros(batch_size, k, latent_dim, device=device, dtype=dtype)


def _fp32_gaussian_latents(
    batch_size: int,
    k: int,
    latent_dim: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    **_: Any,
) -> Tensor:
    """Sample direct H-sized directions with numerically stable FP32 normalization."""

    return normalized_gaussian_latents(
        batch_size,
        k,
        latent_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )


def _engine(
    policy: nn.Module,
    tokenizer: Any,
    config: Mapping[str, Any],
    *,
    max_new_tokens: int,
    seed: int,
    ordinary_sampling: bool = False,
    compute_reference_logprobs: bool = False,
) -> ParticleRolloutEngine:
    particle = config["particles"]
    prompting = config["prompting"]
    generation = config["generation"]
    intervention = getattr(policy, "adapter", None)
    latent_dim = getattr(intervention, "latent_size", None)
    if not isinstance(latent_dim, int) or latent_dim <= 0:
        raise RuntimeError("policy adapter/intervention must expose a positive latent_size")
    if ordinary_sampling:
        latent_sampler = _zero_latents
    elif _particle_mode(config) == "gaussian":
        latent_sampler = _fp32_gaussian_latents
    else:
        latent_sampler = None
    return ParticleRolloutEngine(
        policy,
        tokenizer,
        config=RolloutConfig(
            k=int(particle["count"]),
            latent_dim=latent_dim,
            max_new_tokens=int(max_new_tokens),
            temperature=float(particle["explorer_temperature"]),
            top_p=float(particle["top_p"]),
            response_prefix=str(prompting["response_prefix"]),
            first_token_mode=str(generation.get("first_token_mode", "causal_prefix")),
            condition=str(prompting.get("condition", "synth,cot")),
            use_cache=bool(generation.get("use_cache_for_rollout", True)),
            compute_reference_logprobs=bool(compute_reference_logprobs),
            anchor_greedy=not ordinary_sampling,
            seed=int(seed),
        ),
        latent_sampler=latent_sampler,
    )


def _from_pretrained_bf16(model_class: Any, model_id: str, kwargs: dict[str, Any]) -> nn.Module:
    """Use the native Transformers 5 dtype API, with a narrow compatibility fallback."""

    try:
        return model_class.from_pretrained(model_id, dtype=torch.bfloat16, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc) and "unexpected keyword" not in str(exc):
            raise
        return model_class.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kwargs)


def _load_q_weights(q_head: nn.Module, path: Path) -> None:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("safetensors is required") from exc
    state = load_file(str(path), device=str(next(q_head.parameters()).device))
    missing, unexpected = q_head.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Q checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def load_policy(
    config: Mapping[str, Any],
    device: torch.device,
    *,
    load_warm_q: bool,
    gaussian_scale: float | None = None,
) -> tuple[ParticleHrmForCausalLM, Any]:
    """Load one native HRM replica and the configured intervention plus Q head."""

    try:
        from transformers import AutoTokenizer, HrmTextForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "native HRM-Text support is missing. Install transformers>=5.9,<6, restart "
            "the kernel, then launch this stage in a fresh subprocess."
        ) from exc

    model_config = config["model"]
    model_id = str(model_config["pretrained_model_name_or_path"])
    revision = str(model_config["revision"])
    common = {
        "revision": revision,
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = _from_pretrained_bf16(
        HrmTextForCausalLM,
        model_id,
        {
            **common,
            "attn_implementation": str(model_config.get("attention_implementation", "sdpa")),
            "low_cpu_mem_usage": True,
        },
    ).to(device)

    expected_parameters = int(model_config.get("expected_parameter_count", EXPECTED_MODEL_PARAMETERS))
    actual_parameters = sum(parameter.numel() for parameter in base.parameters())
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"wrong HRM checkpoint/architecture: expected {expected_parameters:,} parameters, "
            f"found {actual_parameters:,}"
        )
    expected_fields = {
        "hidden_size": int(model_config["expected_hidden_size"]),
        "vocab_size": int(model_config["expected_vocab_size"]),
        "H_cycles": int(model_config["expected_h_cycles"]),
        "L_cycles": int(model_config["expected_l_cycles"]),
        # Native Transformers expands num_hidden_layers to recurrence slots and
        # retains the physical per-stack depth in num_layers_per_stack.
        "num_layers_per_stack": int(model_config["expected_layers_per_stack"]),
        "max_position_embeddings": int(model_config["expected_max_position_embeddings"]),
    }
    for field, expected in expected_fields.items():
        observed = getattr(base.config, field, None)
        if observed != expected:
            raise RuntimeError(f"HRM config mismatch: {field}={observed!r}, expected {expected!r}")

    adapter_config = config["adapter"]
    q_config = config["q_warmup"]
    q_head = SharedQHead(
        int(model_config["expected_hidden_size"]),
        int(q_config["bottleneck_size"]),
    )
    q_head.initialize_constant_prior(float(q_config.get("prior_probability", 0.5)))
    mode = _particle_mode(config)
    if mode == "gaussian":
        gaussian_config = config["particles"]["gaussian"]
        selected_scale = (
            float(gaussian_config["default_relative_rms"])
            if gaussian_scale is None
            else float(gaussian_scale)
        )
        intervention: nn.Module = GaussianParticleAdapter(
            GaussianParticleConfig(
                hidden_size=int(model_config["expected_hidden_size"]),
                relative_rms_scale=selected_scale,
                rms_eps=float(adapter_config["rms_epsilon"]),
            )
        )
    else:
        intervention = ParticleAdapter(
            ParticleAdapterConfig(
                hidden_size=int(model_config["expected_hidden_size"]),
                latent_size=int(adapter_config["latent_size"]),
                bottleneck_size=int(adapter_config["bottleneck_size"]),
                initial_relative_rms=float(adapter_config["initial_relative_rms"]),
                max_relative_rms=float(adapter_config["max_relative_rms"]),
                rms_eps=float(adapter_config["rms_epsilon"]),
            )
        )
    policy = ParticleHrmForCausalLM(
        base,
        adapter=intervention,
        q_head=q_head,
        q_bottleneck_size=int(q_config["bottleneck_size"]),
        injection_after_high_cycle=int(adapter_config.get("injection_after_high_cycle", 0)),
        detach_q_state=True,
        freeze_backbone=True,
    )
    if mode == "gaussian":
        # Keep the selected scale buffer FP32; the intervention casts its final
        # delta to the BF16 H-state dtype only after stable normalization.
        policy.adapter.to(device=device)
    else:
        policy.adapter.to(device=device, dtype=torch.bfloat16)
    policy.q_head.to(device=device, dtype=torch.bfloat16)
    for parameter in policy.adapter.parameters():
        parameter.requires_grad_(mode == "learned")
    for parameter in policy.q_head.parameters():
        parameter.requires_grad_(True)
    q_parameters = sum(parameter.numel() for parameter in policy.q_head.parameters())
    if q_parameters != EXPECTED_Q_PARAMETERS:
        raise RuntimeError(f"Q head has {q_parameters:,} parameters; expected {EXPECTED_Q_PARAMETERS:,}")
    if load_warm_q:
        q_path = _project_path(config["paths"]["q_checkpoint"])
        if not q_path.is_file():
            raise FileNotFoundError(f"warm Q checkpoint is missing: {q_path}")
        _load_q_weights(policy.q_head, q_path)
    policy.eval()
    return policy, tokenizer


def _injection_penalty(output: Any, rollout: ParticleRollout, zero: Tensor) -> Tensor:
    values = getattr(output, "relative_rms", None)
    if values is None and isinstance(output, Mapping):
        values = output.get("relative_rms")
    if values is None:
        return zero.sum() * 0.0
    b, k, sequence = rollout.particle_mask.shape
    explorer_positions = rollout.particle_mask.reshape(b * k, sequence).clone()
    explorer_positions.reshape(b, k, sequence)[:, 0] = False
    values = values.float()
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.shape != explorer_positions.shape:
        raise RuntimeError("relative_rms does not align with the replay particle mask")
    if not bool(explorer_positions.any()):
        return zero.sum() * 0.0
    return values[explorer_positions].square().mean()


def _strict_q_labels(rollout: ParticleRollout) -> Tensor:
    labels = rollout.q_labels if rollout.q_labels is not None else rollout.rewards
    if labels.shape != rollout.rewards.shape:
        raise ValueError("q_labels must match rewards [batch,K]")
    labels = labels.float()
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("q_labels must be finite")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("Q is correctness-only: q_labels must be binary {0,1}")
    return labels


class RLTrainModule(nn.Module):
    """One DDP forward containing actor replay and detached-state Q supervision."""

    def __init__(self, policy: ParticleHrmForCausalLM, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.policy = policy
        self.rl_config = dict(config["rl"])

    def forward(self, rollout: ParticleRollout) -> dict[str, Tensor]:
        rewards = rollout.rewards.float()
        labels = _strict_q_labels(rollout)
        advantages = anchor_rescue_advantages(
            rewards, alpha=float(self.rl_config["rescue_alpha"])
        )
        action_mask = rollout.action_mask.clone()
        action_mask[:, 0] = False
        replay = score_actions(self.policy, rollout)
        actor = clipped_token_policy_loss(
            replay.logprobs,
            rollout.old_logprobs,
            advantages,
            action_mask,
            clip_epsilon=float(self.rl_config["ppo_clip_epsilon"]),
            reference_logprobs=rollout.reference_logprobs,
            kl_logprobs=replay.raw_logprobs,
            kl_coefficient=float(self.rl_config["kl_coefficient"]),
        )
        injection = _injection_penalty(replay.output, rollout, replay.logprobs)
        actor_loss = actor.loss + float(
            self.rl_config["injection_penalty_coefficient"]
        ) * injection

        b, k, hidden = rollout.terminal_states.shape
        terminal = rollout.terminal_states.detach().reshape(b * k, hidden)
        prompt = rollout.prompt_summary.detach().reshape(b * k, hidden)
        q_logits = self.policy.q_head(terminal, prompt).reshape(b, k)
        # Keep model weights/activations BF16, but do the numerically sensitive
        # BCE and pairwise softplus reductions in FP32.
        q = supervised_q_loss(
            q_logits.float(),
            labels.float(),
            ranking_weight=float(self.rl_config["q_ranking_weight"]),
        )
        # The parameter sets are disjoint, so summing losses cannot leak Q into
        # actor credit.  It simply lets DDP see both modules in one forward.
        total = actor_loss + q.loss
        return {
            "loss": total,
            "actor_loss": actor_loss,
            "policy_loss": actor.policy_loss,
            "reference_kl": actor.approx_kl,
            "mean_ratio": actor.mean_ratio,
            "clip_fraction": actor.clip_fraction,
            "injection_penalty": injection,
            "q_loss": q.loss,
            "q_bce": q.bce_loss,
            "q_ranking": q.ranking_loss,
            "q_logits": q_logits,
        }


def _trainable_parameters(module: RLTrainModule) -> list[nn.Parameter]:
    parameters = [
        *module.policy.adapter.parameters(),
        *module.policy.q_head.parameters(),
    ]
    ids = {id(parameter) for parameter in parameters}
    unexpected = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and id(parameter) not in ids
    ]
    if unexpected:
        raise RuntimeError("unexpected trainable HRM parameters: " + ", ".join(unexpected))
    if any(parameter.dtype != torch.bfloat16 for parameter in parameters):
        raise RuntimeError("adapter and Q parameters must be BF16")
    return parameters


def _q_logits(policy: ParticleHrmForCausalLM, rollout: ParticleRollout) -> Tensor:
    b, k, hidden = rollout.terminal_states.shape
    return policy.q_head(
        rollout.terminal_states.detach().reshape(b * k, hidden),
        rollout.prompt_summary.detach().reshape(b * k, hidden),
    ).reshape(b, k)


def _parameter_checksum(module: nn.Module) -> Tensor:
    values = [
        value.detach().float().sum()
        for value in (*tuple(module.parameters()), *tuple(module.buffers()))
    ]
    if values:
        return torch.stack(values).sum()
    return torch.tensor(0.0, device=_module_device(module))


def _module_device(module: nn.Module) -> torch.device:
    for value in (*tuple(module.parameters()), *tuple(module.buffers())):
        return value.device
    return torch.device("cpu")


def _destroy_process_group() -> None:
    if dist.is_initialized():
        # Never put a collective in a ``finally`` path.  If one rank raised while
        # its peers are still generating or are inside another collective, a
        # cleanup barrier can deadlock or mismatch the peer collective.
        dist.destroy_process_group()


def _reward_scorers(
    config: Mapping[str, Any], *, need_code: bool
) -> RewardRouter:
    verification = config["verification"]
    math_scorer = MathVerifyScorer()
    code_scorer = None
    if need_code:
        if verification.get("fail_closed_without_sandbox", True) is not True:
            raise ValueError("V1 requires fail_closed_without_sandbox=true")
        expected_commit = str(verification["open_r1_commit"])
        require_installed_vcs_commit("open-r1", expected_commit)
        _validate_code_provider_canary(config)
        code_scorer = OpenR1SandboxCodeScorer(
            provider=str(verification["code_provider"]),
            num_parallel=int(verification["code_parallelism_per_rank"]),
            shape_actor_reward=True,
            binary_reward_weight=float(config["rl"].get("code_binary_reward_weight", 0.8)),
        )
    return RewardRouter(
        math_scorer=math_scorer,
        code_scorer=code_scorer,
        code_binary_weight=float(config["rl"].get("code_binary_reward_weight", 0.8)),
    )


def _validate_code_provider_canary(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require a recent semantic probe before any model-generated code scoring."""

    path = _project_path(config["paths"]["code_provider_canary"])
    if not path.is_file():
        raise FileNotFoundError(
            f"remote code-provider canary is missing: {path}; run v1_code_provider_probe.py"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    verification = config["verification"]
    expected = {
        "format": "hrm-particle-v1-code-provider-canary",
        "provider": str(verification["code_provider"]),
        "harness_version": STRICT_CODE_HARNESS_VERSION,
        "open_r1_commit": str(verification["open_r1_commit"]),
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise RuntimeError("code-provider canary provenance mismatch: " + ", ".join(mismatches))
    checked_at = float(payload.get("checked_at_unix", float("nan")))
    age_seconds = time.time() - checked_at
    if not math.isfinite(age_seconds) or age_seconds < -300 or age_seconds > 48 * 3600:
        raise RuntimeError("code-provider canary is stale or has an invalid timestamp; rerun it")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or float(checks.get("known_good", 0.0)) != 1.0:
        raise RuntimeError("code-provider canary did not record a passing known-good program")
    if any(float(value) != 0.0 for key, value in checks.items() if key != "known_good"):
        raise RuntimeError("code-provider canary recorded a known-bad program as passing")
    return dict(payload)


def _rescore(
    rollout: ParticleRollout,
    examples: Sequence[RolloutExample],
    *,
    router: RewardRouter,
) -> ParticleRollout:
    return rescore_particle_rollout(rollout, examples, router)


def _validated_data_fingerprints(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate prepared data and return hashes of the bytes actually consumed."""

    directory = _project_path(config["paths"]["data_directory"])
    validation = validate_v1_directory(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_manifest_hash = sha256_file(manifest_path)
    if validation.get("manifest_sha256") != actual_manifest_hash:
        raise RuntimeError("prepared-data validation returned the wrong manifest hash")
    declared_files = manifest.get("files")
    if not isinstance(declared_files, Mapping):
        raise RuntimeError("prepared-data manifest has no file mapping")

    files: dict[str, dict[str, Any]] = {}
    for pool in sorted(declared_files):
        info = declared_files[pool]
        if not isinstance(info, Mapping) or not isinstance(info.get("path"), str):
            raise RuntimeError(f"prepared-data manifest entry {pool!r} is malformed")
        file_path = (directory / info["path"]).resolve()
        try:
            relative_path = file_path.relative_to(directory)
        except ValueError as exc:
            raise RuntimeError(f"prepared-data file escapes its directory: {file_path}") from exc
        actual_hash = sha256_file(file_path)
        if actual_hash != info.get("sha256"):
            raise RuntimeError(f"prepared-data checksum mismatch: {file_path.name}")
        files[str(pool)] = {
            "path": str(relative_path),
            "sha256": actual_hash,
            "records": int(info["records"]),
            "bytes": int(file_path.stat().st_size),
        }
    return {
        "manifest_sha256": actual_manifest_hash,
        "schema_version": manifest.get("schema_version"),
        "generator_version": manifest.get("generator_version"),
        "files": files,
    }


def _noise_scale_path(config: Mapping[str, Any]) -> Path:
    return _project_path(config["paths"]["noise_scale_selection"])


def _select_noise_sweep_records(
    records: Sequence[dict[str, Any]],
    *,
    prompt_groups: int,
    code_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, unique development slice from ``rl_train`` only."""

    if prompt_groups <= 0:
        raise ValueError("noise sweep prompt_groups must be positive")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("rl_train contains duplicate prompt IDs")
    math_pool = [record for record in records if record["task_type"] == "math"]
    code_pool = [record for record in records if record["task_type"] == "code"]
    code_groups = math.floor(prompt_groups * code_fraction)
    math_groups = prompt_groups - code_groups
    if len(math_pool) < math_groups or len(code_pool) < code_groups:
        raise ValueError(
            "rl_train is too small for a unique noise-scale development slice: "
            f"need math={math_groups}, code={code_groups}; "
            f"found math={len(math_pool)}, code={len(code_pool)}"
        )
    math_pool = list(math_pool)
    code_pool = list(code_pool)
    random.Random(seed + 11).shuffle(math_pool)
    random.Random(seed + 23).shuffle(code_pool)
    selected = math_pool[:math_groups] + code_pool[:code_groups]
    random.Random(seed + 37).shuffle(selected)
    return selected


def _split_noise_sweep_records(
    records: Sequence[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make deterministic 50/50 halves balanced by task and interleaved by source."""

    selection: list[dict[str, Any]] = []
    confirmation: list[dict[str, Any]] = []
    for task_type in ("math", "code"):
        task_records = [record for record in records if record["task_type"] == task_type]
        if len(task_records) < 2:
            raise ValueError(
                f"noise sweep needs at least two {task_type} prompts for disjoint halves"
            )
        by_source: dict[str, list[dict[str, Any]]] = {}
        for record in task_records:
            by_source.setdefault(str(record.get("source", "")), []).append(record)
        for source, source_records in by_source.items():
            source_records.sort(
                key=lambda record: hashlib.sha256(
                    f"{seed}|{task_type}|{source}|{record['id']}".encode()
                ).digest()
            )
        interleaved: list[dict[str, Any]] = []
        sources = sorted(by_source)
        while len(interleaved) < len(task_records):
            for source in sources:
                if by_source[source]:
                    interleaved.append(by_source[source].pop(0))
        # Give the odd math remainder to selection and the odd code remainder
        # to confirmation.  The default 103/25 mix is therefore exactly 64/64
        # while every task remains balanced to within one prompt.
        if task_type == "math":
            selection.extend(interleaved[::2])
            confirmation.extend(interleaved[1::2])
        else:
            selection.extend(interleaved[1::2])
            confirmation.extend(interleaved[::2])
    return selection, confirmation


def _noise_scale_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    memory_path = _memory_plan_path(config)
    if not memory_path.is_file():
        raise FileNotFoundError(
            f"VRAM plan is missing: {memory_path}; run vram_plan before noise_scale_sweep"
        )
    records = _load_records(
        _project_path(config["paths"]["data_directory"]) / "rl_train.jsonl"
    )
    gaussian = config["particles"]["gaussian"]
    selected = _select_noise_sweep_records(
        records,
        prompt_groups=int(gaussian["sweep_prompt_groups"]),
        code_fraction=float(config["rl"]["code_fraction"]),
        seed=int(config["seed"]) + 271,
    )
    selected_ids = [str(record["id"]) for record in selected]
    sweep_config = {
        "seed": config["seed"],
        "model": config["model"],
        "prompting": config["prompting"],
        "adapter_injection": {
            "rms_epsilon": config["adapter"]["rms_epsilon"],
            "injection_after_high_cycle": config["adapter"]["injection_after_high_cycle"],
        },
        "particles": config["particles"],
        "generation": {
            "q_collect_max_new_tokens": config["generation"]["q_collect_max_new_tokens"],
            "first_token_mode": config["generation"]["first_token_mode"],
            "use_cache_for_rollout": config["generation"]["use_cache_for_rollout"],
        },
        "code_fraction": config["rl"]["code_fraction"],
        "verification": config["verification"],
        "world_size": config["runtime"]["world_size"],
    }
    return {
        "particle_mode": "gaussian",
        "sweep_config_hash": _canonical_hash(sweep_config),
        "model_revision": str(config["model"]["revision"]),
        "data": _validated_data_fingerprints(config),
        "memory_plan_sha256": sha256_file(memory_path),
        "selected_prompt_ids_sha256": _canonical_hash(selected_ids),
        "selected_prompt_groups": len(selected_ids),
        "code_hash": _code_hash(),
        "package_versions": package_versions(),
    }


def _load_noise_scale_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    if _particle_mode(config) != "gaussian":
        raise ValueError("noise-scale selection exists only in gaussian particle mode")
    path = _noise_scale_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"noise-scale selection is missing: {path}; run noise_scale_sweep first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("noise-scale selection root is not an object")
    if (
        payload.get("format") != NOISE_SCALE_FORMAT
        or payload.get("version") != NOISE_SCALE_VERSION
    ):
        raise RuntimeError("unsupported noise-scale selection artifact")
    if payload.get("identity") != _noise_scale_identity(config):
        raise RuntimeError(
            "noise-scale selection does not match this config/model/data/code"
        )
    selected = float(payload.get("selected_relative_rms", float("nan")))
    positive_candidates = {
        float(value)
        for value in config["particles"]["gaussian"]["scale_candidates"]
        if float(value) > 0.0
    }
    if not math.isfinite(selected) or selected not in positive_candidates:
        raise RuntimeError("noise-scale artifact selected an undeclared/non-positive scale")
    return payload


def _scale_rows_by_prompt(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[float],
    k: int,
) -> dict[float, dict[str, dict[str, Any]]]:
    expected_scales = [float(value) for value in candidates]
    by_scale: dict[float, dict[str, dict[str, Any]]] = {
        scale: {} for scale in expected_scales
    }
    for raw in rows:
        scale = float(raw.get("scale", float("nan")))
        if scale not in by_scale:
            raise RuntimeError(f"noise sweep row has undeclared scale {scale!r}")
        prompt_id = str(raw.get("id", ""))
        if not prompt_id or prompt_id in by_scale[scale]:
            raise RuntimeError("noise sweep has a missing or duplicate id/scale cell")
        correctness = raw.get("correctness")
        if not isinstance(correctness, list) or len(correctness) != k:
            raise RuntimeError(f"noise sweep correctness must contain exactly K={k} labels")
        numeric = [float(value) for value in correctness]
        if any(value not in {0.0, 1.0} for value in numeric):
            raise RuntimeError("noise sweep correctness labels must be binary")
        by_scale[scale][prompt_id] = {**dict(raw), "correctness": numeric}
    baseline_ids = set(by_scale[expected_scales[0]])
    if not baseline_ids:
        raise RuntimeError("noise sweep contains no prompt groups")
    for scale, scale_rows in by_scale.items():
        if set(scale_rows) != baseline_ids:
            raise RuntimeError(f"noise sweep prompt IDs are incomplete at scale {scale}")
    return by_scale


def _aggregate_noise_scale_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[float],
    k: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Validate paired sweep rows and compute correctness-only scale metrics."""

    candidates = [float(value) for value in candidates]
    baseline_scale = 0.0
    if (
        not candidates
        or candidates[0] != baseline_scale
        or candidates != sorted(set(candidates))
    ):
        raise RuntimeError(
            "noise sweep candidates must be sorted, unique, and begin with zero"
        )
    if k < 2:
        raise RuntimeError("noise sweep K must include an anchor and explorer")
    by_scale = _scale_rows_by_prompt(rows, candidates=candidates, k=k)
    prompt_ids = sorted(by_scale[baseline_scale])
    baseline_anchor = {
        prompt_id: by_scale[baseline_scale][prompt_id]["correctness"][0]
        for prompt_id in prompt_ids
    }
    baseline_anchor_text = {
        prompt_id: (
            by_scale[baseline_scale][prompt_id].get("response_texts") or [None]
        )[0]
        for prompt_id in prompt_ids
    }
    baseline_oracle = [
        max(by_scale[baseline_scale][prompt_id]["correctness"])
        for prompt_id in prompt_ids
    ]
    metrics: dict[str, Any] = {}
    for scale_index, scale in enumerate(candidates):
        scale_rows = by_scale[scale]
        for prompt_id in prompt_ids:
            if scale_rows[prompt_id]["correctness"][0] != baseline_anchor[prompt_id]:
                raise RuntimeError(
                    "greedy branch-zero correctness changed across Gaussian scales"
                )
            response_texts = scale_rows[prompt_id].get("response_texts") or [None]
            if response_texts[0] != baseline_anchor_text[prompt_id]:
                raise RuntimeError(
                    "greedy branch-zero text changed across Gaussian scales"
                )
        labels = [scale_rows[prompt_id]["correctness"] for prompt_id in prompt_ids]
        oracle = [max(group) for group in labels]
        explorer = [sum(group[1:]) / (k - 1) for group in labels]
        mixed = [0.0 < sum(group) < k for group in labels]
        rescue = [group[0] == 0.0 and max(group[1:]) == 1.0 for group in labels]
        interval = paired_bootstrap_delta(
            oracle,
            baseline_oracle,
            num_samples=bootstrap_samples,
            seed=seed + scale_index * 101,
        )
        task_metrics: dict[str, Any] = {}
        task_types = sorted(
            {str(scale_rows[prompt_id]["task_type"]) for prompt_id in prompt_ids}
        )
        for task_type in task_types:
            indices = [
                index
                for index, prompt_id in enumerate(prompt_ids)
                if str(scale_rows[prompt_id]["task_type"]) == task_type
            ]
            task_metrics[task_type] = {
                "prompt_groups": len(indices),
                "anchor_accuracy": sum(labels[index][0] for index in indices) / len(indices),
                "oracle_at_k": sum(oracle[index] for index in indices) / len(indices),
                "explorer_mean_accuracy": sum(explorer[index] for index in indices)
                / len(indices),
                "mixed_fraction": sum(mixed[index] for index in indices) / len(indices),
                "rescue_fraction": sum(rescue[index] for index in indices) / len(indices),
            }
        metrics[str(scale)] = {
            "scale": scale,
            "prompt_groups": len(prompt_ids),
            "anchor_accuracy": sum(group[0] for group in labels) / len(labels),
            "oracle_at_k": sum(oracle) / len(oracle),
            "explorer_mean_accuracy": sum(explorer) / len(explorer),
            "mixed_fraction": sum(mixed) / len(mixed),
            "rescue_fraction": sum(rescue) / len(rescue),
            "oracle_delta_vs_zero": {
                "mean_delta": interval.mean_delta,
                "low": interval.low,
                "high": interval.high,
                "confidence": interval.confidence,
            },
            "by_task": task_metrics,
        }
    return {"prompt_ids": prompt_ids, "metrics": metrics}


def _select_noise_scale(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    candidates: Sequence[float],
    tie_tolerance: float,
) -> float:
    positive = sorted(float(value) for value in candidates if float(value) > 0.0)
    if not positive:
        raise ValueError("noise-scale selection requires a positive candidate")
    baseline = float(metrics["0.0"]["oracle_at_k"])
    best = max(float(metrics[str(scale)]["oracle_at_k"]) for scale in positive)
    if best + tie_tolerance < baseline:
        raise RuntimeError(
            "every positive Gaussian scale underperformed the zero-noise control on "
            "the scale-selection split; refusing to force a harmful particle"
        )
    eligible = [
        scale
        for scale in positive
        if best - float(metrics[str(scale)]["oracle_at_k"]) <= tie_tolerance
    ]
    return min(eligible)


def _q_state_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    directory = _project_path(config["paths"]["q_state_directory"])
    return directory, _project_path(config["paths"]["q_checkpoint"])


def _initial_adapter_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    directory, _ = _q_state_paths(config)
    return directory / "initial-adapter.safetensors", directory / "initial-adapter.json"


def _initial_adapter_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    collection_config = {
        "seed": config["seed"],
        "model": config["model"],
        "prompting": config["prompting"],
        "adapter": config["adapter"],
        "particles": config["particles"],
        "verification": config["verification"],
        "q_collect_max_new_tokens": config["generation"]["q_collect_max_new_tokens"],
        "first_token_mode": config["generation"]["first_token_mode"],
        "use_cache_for_rollout": config["generation"]["use_cache_for_rollout"],
        "world_size": config["runtime"]["world_size"],
    }
    identity = {
        "model_id": config["model"]["pretrained_model_name_or_path"],
        "model_revision": config["model"]["revision"],
        "collection_config_hash": _canonical_hash(collection_config),
        "data": _validated_data_fingerprints(config),
        "code_hash": _code_hash(),
        "package_versions": package_versions(),
    }
    if _particle_mode(config) == "gaussian":
        selection = _load_noise_scale_selection(config)
        selection_path = _noise_scale_path(config)
        identity["noise_scale_selection_sha256"] = sha256_file(selection_path)
        identity["selected_relative_rms"] = float(selection["selected_relative_rms"])
    return identity


def _broadcast_module(module: nn.Module, context: DistributedContext) -> None:
    if context.world_size == 1:
        return
    with torch.no_grad():
        for parameter in module.parameters():
            dist.broadcast(parameter, src=0)
        for buffer in module.buffers():
            dist.broadcast(buffer, src=0)


def _synchronize_small_modules(
    policy: ParticleHrmForCausalLM, context: DistributedContext
) -> None:
    """Synchronize only the adapter/Q state, never the frozen 1B backbone."""

    _broadcast_module(policy.adapter, context)
    _broadcast_module(policy.q_head, context)


def _atomic_safetensors_save(state: Mapping[str, Tensor], path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("safetensors is required") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    try:
        save_file(dict(state), str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_initial_adapter_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    tensor_path, metadata_path = _initial_adapter_paths(config)
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "initial adapter artifact is missing; rerun the collect_q stage before Q/RL"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        payload.get("format") != INITIAL_ADAPTER_FORMAT
        or payload.get("version") != INITIAL_ADAPTER_VERSION
    ):
        raise RuntimeError("unsupported initial adapter artifact")
    if payload.get("tensor_file") != tensor_path.name:
        raise RuntimeError("initial adapter metadata points to the wrong tensor file")
    actual_hash = sha256_file(tensor_path)
    if payload.get("sha256") != actual_hash:
        raise RuntimeError("initial adapter checksum failed")
    expected_identity = _initial_adapter_identity(config)
    if payload.get("identity") != expected_identity:
        raise RuntimeError("initial adapter provenance does not match this run")
    return dict(payload)


def _load_initial_adapter_artifact(
    adapter: nn.Module, config: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = _read_initial_adapter_metadata(config)
    tensor_path, _ = _initial_adapter_paths(config)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("safetensors is required") from exc
    device = str(_module_device(adapter))
    state = load_file(str(tensor_path), device=device)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"initial adapter checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return metadata


def _save_initial_adapter_artifact(
    adapter: nn.Module,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> dict[str, Any]:
    tensor_path, metadata_path = _initial_adapter_paths(config)
    if context.is_main:
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in adapter.state_dict().items()
        }
        _atomic_safetensors_save(state, tensor_path)
        _atomic_json(
            metadata_path,
            {
                "format": INITIAL_ADAPTER_FORMAT,
                "version": INITIAL_ADAPTER_VERSION,
                "tensor_file": tensor_path.name,
                "sha256": sha256_file(tensor_path),
                "identity": _initial_adapter_identity(config),
            },
        )
    _barrier(context)
    # Loading the just-written artifact on every rank both verifies its checksum
    # and makes the collection behavior byte-identical to the persisted actor.
    return _load_initial_adapter_artifact(adapter, config)


def stage_noise_scale_sweep(config: Mapping[str, Any]) -> None:
    """Select a fixed direct-Gaussian RMS scale on a disjoint RL dev slice."""

    if _particle_mode(config) != "gaussian":
        raise ValueError("noise_scale_sweep is valid only for particles.mode='gaussian'")
    context = _distributed_context(config, require_distributed=True)
    try:
        _seed_everything(int(config["seed"]) + 271, context)
        memory_plan = _load_memory_plan(config, context)
        destination = _noise_scale_path(config)
        if destination.exists():
            payload = _load_noise_scale_selection(config)
            if context.is_main:
                print(
                    json.dumps(
                        {
                            "noise_scale_sweep": "reused",
                            "selected_relative_rms": payload["selected_relative_rms"],
                            "artifact": str(destination),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return

        gaussian = config["particles"]["gaussian"]
        candidates = [float(value) for value in gaussian["scale_candidates"]]
        records = _load_records(
            _project_path(config["paths"]["data_directory"]) / "rl_train.jsonl"
        )
        selected_records = _select_noise_sweep_records(
            records,
            prompt_groups=int(gaussian["sweep_prompt_groups"]),
            code_fraction=float(config["rl"]["code_fraction"]),
            seed=int(config["seed"]) + 271,
        )
        local_records = selected_records[context.rank :: context.world_size]
        router = _reward_scorers(
            config,
            need_code=any(record["task_type"] == "code" for record in local_records),
        )
        policy, tokenizer = load_policy(
            config,
            context.device,
            load_warm_q=False,
            gaussian_scale=float(gaussian["default_relative_rms"]),
        )
        if not isinstance(policy.adapter, GaussianParticleAdapter):
            raise RuntimeError("gaussian mode constructed the wrong intervention")
        _synchronize_small_modules(policy, context)

        rows: list[dict[str, Any]] = []
        prompt_batch_size = int(memory_plan["rollout_prompt_batch_size"])
        # Batch-outer/scale-inner is deliberate.  A fresh engine for every
        # batch/scale pair gives all scales the same latent and token RNG state,
        # even when a previous scale reaches EOS earlier than another.
        for batch_index, record_batch in enumerate(
            _chunks(local_records, prompt_batch_size)
        ):
            examples = [
                _rollout_example(record, config["prompting"])
                for record in record_batch
            ]
            common_seed = (
                int(config["seed"])
                + 10_007
                + context.rank * 100_003
                + batch_index * 1_009
            )
            for scale in candidates:
                policy.adapter.set_relative_rms_scale(scale)
                engine = _engine(
                    policy,
                    tokenizer,
                    config,
                    max_new_tokens=int(
                        config["generation"]["q_collect_max_new_tokens"]
                    ),
                    seed=common_seed,
                    compute_reference_logprobs=False,
                )
                rollout = _rescore(
                    engine.generate(examples), examples, router=router
                )
                correctness = _strict_q_labels(rollout).cpu()
                rewards = rollout.rewards.detach().float().cpu()
                for row_index, record in enumerate(record_batch):
                    rows.append(
                        {
                            "id": str(record["id"]),
                            "task_type": str(record["task_type"]),
                            "source": str(record.get("source", "")),
                            "scale": scale,
                            "common_rng_seed": common_seed,
                            "correctness": [
                                int(value) for value in correctness[row_index].tolist()
                            ],
                            "actor_rewards": [
                                float(value) for value in rewards[row_index].tolist()
                            ],
                            "response_texts": list(rollout.response_texts[row_index]),
                        }
                    )
            if context.is_main:
                print(
                    json.dumps(
                        {
                            "noise_sweep_groups_rank0": min(
                                (batch_index + 1) * prompt_batch_size,
                                len(local_records),
                            ),
                            "noise_scales": candidates,
                        }
                    ),
                    flush=True,
                )

        state_directory = (
            _project_path(config["paths"]["run_directory"]) / "noise_scale_sweep"
        )
        shard = state_directory / f"rank-{context.rank:02d}.jsonl"
        _write_jsonl_records(shard, rows)
        _barrier(context)
        if context.is_main:
            merged: list[dict[str, Any]] = []
            shard_hashes: dict[str, str] = {}
            shard_paths = sorted(state_directory.glob("rank-*.jsonl"))
            if len(shard_paths) != context.world_size:
                raise RuntimeError(
                    f"noise sweep found {len(shard_paths)} shards; "
                    f"expected {context.world_size}"
                )
            for shard_path in shard_paths:
                shard_hashes[shard_path.name] = sha256_file(shard_path)
                with shard_path.open("r", encoding="utf-8") as handle:
                    merged.extend(
                        json.loads(line) for line in handle if line.strip()
                    )
            expected_rows = len(selected_records) * len(candidates)
            if len(merged) != expected_rows:
                raise RuntimeError(
                    f"noise sweep produced {len(merged)} rows; expected {expected_rows}"
                )
            aggregate = _aggregate_noise_scale_rows(
                merged,
                candidates=candidates,
                k=int(config["particles"]["count"]),
                bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["seed"]) + 12_019,
            )
            selection_records, confirmation_records = _split_noise_sweep_records(
                selected_records, seed=int(config["seed"]) + 12_023
            )
            selection_ids = {str(record["id"]) for record in selection_records}
            confirmation_ids = {
                str(record["id"]) for record in confirmation_records
            }
            selection_aggregate = _aggregate_noise_scale_rows(
                [row for row in merged if str(row["id"]) in selection_ids],
                candidates=candidates,
                k=int(config["particles"]["count"]),
                bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["seed"]) + 12_029,
            )
            confirmation_aggregate = _aggregate_noise_scale_rows(
                [row for row in merged if str(row["id"]) in confirmation_ids],
                candidates=candidates,
                k=int(config["particles"]["count"]),
                bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["seed"]) + 12_037,
            )
            selected_scale = _select_noise_scale(
                selection_aggregate["metrics"],
                candidates=candidates,
                tie_tolerance=float(gaussian["oracle_tie_tolerance"]),
            )
            selected_metric = confirmation_aggregate["metrics"][str(selected_scale)]
            confirmation_delta = float(
                selected_metric["oracle_delta_vs_zero"]["mean_delta"]
            )
            tolerance = float(gaussian["oracle_tie_tolerance"])
            confirmation_task_deltas = {
                task_type: (
                    float(selected_metric["by_task"][task_type]["oracle_at_k"])
                    - float(
                        confirmation_aggregate["metrics"]["0.0"]["by_task"][
                            task_type
                        ]["oracle_at_k"]
                    )
                )
                for task_type in ("math", "code")
            }
            if confirmation_delta + tolerance < 0.0 or any(
                delta + tolerance < 0.0
                for delta in confirmation_task_deltas.values()
            ):
                raise RuntimeError(
                    "the selected positive Gaussian scale underperformed zero noise on "
                    "the disjoint confirmation split overall or within a task; "
                    "refusing to freeze it"
                )
            evidence_positive = (
                float(selected_metric["oracle_delta_vs_zero"]["low"]) > 0.0
            )
            payload = {
                "format": NOISE_SCALE_FORMAT,
                "version": NOISE_SCALE_VERSION,
                "identity": _noise_scale_identity(config),
                "selected_relative_rms": selected_scale,
                "development_evidence_positive": evidence_positive,
                "selection_rule": (
                    "smallest positive scale within oracle_tie_tolerance of the "
                    "best oracle@4 on the selection half, followed by a disjoint "
                    "confirmation no-degradation gate; scale zero is baseline-only"
                ),
                "selection_objective": "strict binary correctness oracle@4",
                "confirmation_gate": (
                    "point-estimate no-degradation overall and separately for math/code"
                ),
                "confirmation_task_deltas": confirmation_task_deltas,
                "scale_candidates": candidates,
                "oracle_tie_tolerance": float(gaussian["oracle_tie_tolerance"]),
                "prompt_ids": aggregate["prompt_ids"],
                "prompt_groups": len(aggregate["prompt_ids"]),
                "metrics": aggregate["metrics"],
                "selection_prompt_ids": selection_aggregate["prompt_ids"],
                "confirmation_prompt_ids": confirmation_aggregate["prompt_ids"],
                "selection_metrics": selection_aggregate["metrics"],
                "confirmation_metrics": confirmation_aggregate["metrics"],
                "shard_sha256": shard_hashes,
                "rng_design": (
                    "fresh common latent/token RNG seed for every batch across scales"
                ),
                "note": (
                    "Selected only on a deterministic unique rl_train development slice. "
                    "Task-balanced, source-interleaved prompt IDs form disjoint selection "
                    "and confirmation halves; the evidence flag is computed only on "
                    "confirmation. "
                    "No q_warm or final benchmark examples were used. Actor rewards are "
                    "logged for audit but never choose the scale."
                ),
            }
            _atomic_json(destination, payload)
            print(
                json.dumps(
                    {
                        "noise_scale_sweep": "complete",
                        "selected_relative_rms": selected_scale,
                        "development_evidence_positive": evidence_positive,
                        "artifact": str(destination),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _barrier(context)
        _load_noise_scale_selection(config)
    finally:
        _destroy_process_group()


def stage_collect_q(config: Mapping[str, Any]) -> None:
    """Collect exactly K candidate states per prepared prompt, sharded by rank."""

    context = _distributed_context(config, require_distributed=True)
    try:
        _seed_everything(int(config["seed"]), context)
        memory_plan = _load_memory_plan(config, context)
        prompt_batch_size = int(memory_plan["rollout_prompt_batch_size"])
        memory_plan_sha256 = sha256_file(_memory_plan_path(config))
        data_fingerprints = _validated_data_fingerprints(config)
        scale_selection = (
            _load_noise_scale_selection(config)
            if _particle_mode(config) == "gaussian"
            else None
        )
        data_path = _project_path(config["paths"]["data_directory"]) / "q_warm.jsonl"
        records = _load_records(data_path)
        expected_groups = int(config["q_warmup"]["expected_prompt_groups"])
        if len(records) != expected_groups:
            raise RuntimeError(
                f"Q collection expects {expected_groups} prompt groups, found {len(records)}"
            )
        local_records = records[context.rank :: context.world_size]
        need_code = any(record["task_type"] == "code" for record in local_records)
        router = _reward_scorers(config, need_code=need_code)
        policy, tokenizer = load_policy(
            config,
            context.device,
            load_warm_q=False,
            gaussian_scale=(
                float(scale_selection["selected_relative_rms"])
                if scale_selection is not None
                else None
            ),
        )
        # Rank-local model construction consumes rank-local RNG.  Explicitly
        # synchronize the two small modules, then persist the exact adapter that
        # generated Q warmup candidates for reuse by pilot/full RL.
        _synchronize_small_modules(policy, context)
        initial_adapter = _save_initial_adapter_artifact(policy.adapter, config, context)
        engine = _engine(
            policy,
            tokenizer,
            config,
            max_new_tokens=int(config["generation"]["q_collect_max_new_tokens"]),
            seed=int(config["seed"]) + context.rank * 100_003,
            compute_reference_logprobs=False,
        )

        terminal_groups: list[Tensor] = []
        prompt_groups: list[Tensor] = []
        label_groups: list[Tensor] = []
        actor_reward_groups: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        processed = 0
        for record_batch in _chunks(local_records, prompt_batch_size):
            examples = [
                _rollout_example(record, config["prompting"])
                for record in record_batch
            ]
            rollout = engine.generate(examples)
            rollout = _rescore(
                rollout,
                examples,
                router=router,
            )
            labels = _strict_q_labels(rollout)
            for batch_index, record in enumerate(record_batch):
                terminal_groups.append(
                    rollout.terminal_states[batch_index]
                    .detach()
                    .cpu()
                    .to(torch.bfloat16)
                )
                prompt_groups.append(
                    rollout.prompt_summary[batch_index]
                    .detach()
                    .cpu()
                    .to(torch.bfloat16)
                )
                label_groups.append(labels[batch_index].detach().cpu())
                actor_reward_groups.append(
                    rollout.rewards[batch_index].detach().cpu().float()
                )
                metadata.append(
                    {
                        "id": record["id"],
                        "task_type": record["task_type"],
                        "source": record.get("source"),
                        "candidate_correctness": [
                            int(value) for value in labels[batch_index].tolist()
                        ],
                    }
                )
            processed += len(record_batch)
            if context.is_main and processed % 25 < len(record_batch):
                print(json.dumps({"q_collection_groups_rank0": processed}), flush=True)

        tensors = {
            "terminal_states": torch.stack(terminal_groups).contiguous(),
            "prompt_summaries": torch.stack(prompt_groups).contiguous(),
            "correctness": torch.stack(label_groups).float().contiguous(),
            "actor_rewards": torch.stack(actor_reward_groups).float().contiguous(),
        }
        directory, _ = _q_state_paths(config)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required") from exc
        shard = directory / f"q-states-rank-{context.rank:02d}.safetensors"
        save_file(tensors, str(shard))
        _atomic_json(
            directory / f"q-states-rank-{context.rank:02d}.json",
            {
                "rank": context.rank,
                "world_size": context.world_size,
                "model_revision": config["model"]["revision"],
                "initial_adapter_sha256": initial_adapter["sha256"],
                "noise_scale_selection_sha256": (
                    sha256_file(_noise_scale_path(config))
                    if scale_selection is not None
                    else None
                ),
                "memory_plan_sha256": memory_plan_sha256,
                "data_fingerprints": data_fingerprints,
                "groups": metadata,
                "tensor_file": shard.name,
                "tensor_sha256": sha256_file(shard),
            },
        )
        local_candidates = torch.tensor(
            tensors["correctness"].numel(), device=context.device, dtype=torch.long
        )
        if context.world_size > 1:
            dist.all_reduce(local_candidates, op=dist.ReduceOp.SUM)
        expected_candidates = int(config["q_warmup"]["target_candidate_examples"])
        if int(local_candidates) != expected_candidates:
            raise RuntimeError(
                f"collected {int(local_candidates)} Q candidates; expected {expected_candidates}"
            )
        _barrier(context)
        if context.is_main:
            _atomic_json(
                directory / "collection-complete.json",
                {
                    "candidate_examples": int(local_candidates),
                    "prompt_groups": expected_groups,
                    "k": int(config["particles"]["count"]),
                    "world_size": context.world_size,
                    "model_revision": config["model"]["revision"],
                    "initial_adapter_sha256": initial_adapter["sha256"],
                    "noise_scale_selection_sha256": (
                        sha256_file(_noise_scale_path(config))
                        if scale_selection is not None
                        else None
                    ),
                    "memory_plan_sha256": memory_plan_sha256,
                    "data_fingerprints": data_fingerprints,
                    "note": "Q labels are strict external correctness, never Q self-reward.",
                },
            )
    finally:
        _destroy_process_group()


def _load_q_state_shards(
    config: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor, list[str], list[dict[str, Any]]]:
    directory, _ = _q_state_paths(config)
    completion = directory / "collection-complete.json"
    if not completion.is_file():
        raise FileNotFoundError("Q state collection is incomplete")
    completion_payload = json.loads(completion.read_text(encoding="utf-8"))
    initial_adapter = _read_initial_adapter_metadata(config)
    data_fingerprints = _validated_data_fingerprints(config)
    memory_plan_sha256 = sha256_file(_memory_plan_path(config))
    noise_scale_sha256 = (
        sha256_file(_noise_scale_path(config))
        if _particle_mode(config) == "gaussian"
        else None
    )
    expected_shards = int(config["runtime"]["world_size"])
    if int(completion_payload.get("world_size", -1)) != expected_shards:
        raise RuntimeError("Q state completion world size does not match the config")
    if completion_payload.get("model_revision") != config["model"]["revision"]:
        raise RuntimeError("Q state completion model revision changed")
    if completion_payload.get("initial_adapter_sha256") != initial_adapter["sha256"]:
        raise RuntimeError("Q state collection does not match the persisted initial adapter")
    if completion_payload.get("data_fingerprints") != data_fingerprints:
        raise RuntimeError("Q state collection does not match the prepared data")
    if completion_payload.get("memory_plan_sha256") != memory_plan_sha256:
        raise RuntimeError("Q state collection does not match the VRAM batch plan")
    if completion_payload.get("noise_scale_selection_sha256") != noise_scale_sha256:
        raise RuntimeError("Q state collection used a different noise-scale selection")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    terminals: list[Tensor] = []
    prompts: list[Tensor] = []
    labels: list[Tensor] = []
    ids: list[str] = []
    groups: list[dict[str, Any]] = []
    tensor_paths = sorted(directory.glob("q-states-rank-*.safetensors"))
    if len(tensor_paths) != expected_shards:
        raise RuntimeError(
            f"Q state collection has {len(tensor_paths)} shards; expected {expected_shards}"
        )
    for expected_rank, tensor_path in enumerate(tensor_paths):
        sidecar = tensor_path.with_suffix(".json")
        audit = json.loads(sidecar.read_text(encoding="utf-8"))
        if int(audit.get("rank", -1)) != expected_rank:
            raise RuntimeError(f"Q state shard rank metadata is wrong: {tensor_path}")
        if int(audit.get("world_size", -1)) != expected_shards:
            raise RuntimeError(f"Q state shard world size is wrong: {tensor_path}")
        if audit.get("model_revision") != config["model"]["revision"]:
            raise RuntimeError(f"Q state shard model revision changed: {tensor_path}")
        if audit.get("tensor_file") != tensor_path.name:
            raise RuntimeError(f"Q state sidecar points to the wrong tensor: {tensor_path}")
        if sha256_file(tensor_path) != audit["tensor_sha256"]:
            raise RuntimeError(f"Q state shard checksum failed: {tensor_path}")
        if audit.get("initial_adapter_sha256") != initial_adapter["sha256"]:
            raise RuntimeError(f"Q state shard used a different initial adapter: {tensor_path}")
        if audit.get("data_fingerprints") != data_fingerprints:
            raise RuntimeError(f"Q state shard used different prepared data: {tensor_path}")
        if audit.get("memory_plan_sha256") != memory_plan_sha256:
            raise RuntimeError(f"Q state shard used a different VRAM plan: {tensor_path}")
        if audit.get("noise_scale_selection_sha256") != noise_scale_sha256:
            raise RuntimeError(
                f"Q state shard used a different noise-scale selection: {tensor_path}"
            )
        payload = load_file(str(tensor_path), device="cpu")
        terminals.append(payload["terminal_states"])
        prompts.append(payload["prompt_summaries"])
        labels.append(payload["correctness"].float())
        shard_groups = audit.get("groups")
        if not isinstance(shard_groups, list) or len(shard_groups) != payload["correctness"].shape[0]:
            raise RuntimeError(f"Q state sidecar groups do not align: {tensor_path}")
        for group in shard_groups:
            if not isinstance(group, dict) or group.get("task_type") not in {"math", "code"}:
                raise RuntimeError(f"Q state sidecar has malformed group metadata: {tensor_path}")
            groups.append(dict(group))
            ids.append(str(group["id"]))
    if not terminals:
        raise FileNotFoundError(f"no Q state shards found in {directory}")
    terminal = torch.cat(terminals)
    prompt = torch.cat(prompts)
    correctness = torch.cat(labels)
    if not (terminal.shape == prompt.shape and terminal.shape[:2] == correctness.shape):
        raise RuntimeError("Q state shard tensor shapes do not align")
    if len(ids) != terminal.shape[0]:
        raise RuntimeError("Q sidecar group IDs do not align with tensors")
    expected = int(config["q_warmup"]["target_candidate_examples"])
    if correctness.numel() != expected:
        raise RuntimeError(f"loaded {correctness.numel()} Q candidates; expected {expected}")
    if len(groups) != terminal.shape[0]:
        raise RuntimeError("Q group metadata does not align with tensors")
    prepared_ids = {
        str(record["id"])
        for record in _load_records(
            _project_path(config["paths"]["data_directory"]) / "q_warm.jsonl"
        )
    }
    if len(ids) != len(set(ids)) or set(ids) != prepared_ids:
        raise RuntimeError(
            "Q state shard prompt IDs are duplicated or do not match prepared q_warm"
        )
    return terminal, prompt, correctness, ids, groups


@torch.no_grad()
def _score_q_groups(
    q_head: SharedQHead,
    terminal: Tensor,
    prompt: Tensor,
    *,
    group_batch: int,
    device: torch.device,
) -> Tensor:
    groups, k, hidden = terminal.shape
    outputs: list[Tensor] = []
    q_head.eval()
    for start in range(0, groups, group_batch):
        stop = min(start + group_batch, groups)
        t = terminal[start:stop].to(device=device, dtype=torch.bfloat16)
        p = prompt[start:stop].to(device=device, dtype=torch.bfloat16)
        outputs.append(q_head(t.reshape(-1, hidden), p.reshape(-1, hidden)).reshape(-1, k).float().cpu())
    return torch.cat(outputs)


def stage_q_warmup(config: Mapping[str, Any]) -> None:
    """Train only Q on detached cached states; the 1B policy is not loaded."""

    context = _distributed_context(config, require_distributed=False)
    if context.world_size != 1:
        raise RuntimeError("Q warmup is intentionally a one-GPU stage")
    memory_plan = _load_memory_plan(config, context)
    _seed_everything(int(config["seed"]) + 71, context)
    terminal, prompt, correctness, ids, groups = _load_q_state_shards(config)
    split_percentages = config["q_warmup"]["split_percentages"]
    masks = stratified_prompt_split_masks(
        ids,
        [str(group.get("source", "")) for group in groups],
        split_percentages,
    )
    indices = {name: torch.where(mask)[0] for name, mask in masks.items()}
    train_indices = indices["train"]
    early_stop_indices = indices["early_stop"]
    margin_select_indices = indices["margin_select"]
    safety_test_indices = indices["safety_test"]
    minimum = int(config["q_warmup"]["minimum_global_calibration_prompts"])
    for name in ("early_stop", "margin_select", "safety_test"):
        if len(indices[name]) < minimum:
            raise RuntimeError(
                f"Q split {name!r} has {len(indices[name])} prompts; need at least {minimum}"
            )
    positive_rate = float(correctness[train_indices].mean())
    if not 0.0 < positive_rate < 1.0:
        raise RuntimeError(
            f"Q warmup has degenerate correctness prevalence {positive_rate:.4f}; "
            "change the prompt difficulty mix before training"
        )
    prior = min(0.95, max(0.05, positive_rate))
    hidden = terminal.shape[-1]
    q_head = SharedQHead(hidden, int(config["q_warmup"]["bottleneck_size"]))
    q_head.initialize_constant_prior(prior)
    q_head.to(device=context.device, dtype=torch.bfloat16)
    optimizer = BF16MasterAdamW(
        q_head.parameters(),
        lr=float(config["q_warmup"]["learning_rate"]),
        weight_decay=float(config["q_warmup"]["weight_decay"]),
    )
    batch_groups = int(memory_plan["q_batch_prompt_groups"])
    epochs = int(config["q_warmup"]["epochs"])
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 991)
    best_bce = float("inf")
    best_state: dict[str, Tensor] | None = None
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        q_head.train()
        losses: list[float] = []
        for start in range(0, len(permutation), batch_groups):
            group_indices = permutation[start : start + batch_groups]
            t = terminal[group_indices].to(context.device, dtype=torch.bfloat16)
            p = prompt[group_indices].to(context.device, dtype=torch.bfloat16)
            y = correctness[group_indices].to(context.device).float()
            b, k, h = t.shape
            logits = q_head(t.reshape(b * k, h), p.reshape(b * k, h)).reshape(b, k)
            loss = supervised_q_loss(
                logits.float(),
                y.float(),
                ranking_weight=float(config["q_warmup"]["ranking_weight"]),
            ).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(q_head.parameters()), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        early_stop_logits = _score_q_groups(
            q_head,
            terminal[early_stop_indices],
            prompt[early_stop_indices],
            group_batch=batch_groups,
            device=context.device,
        )
        early_stop_labels = correctness[early_stop_indices].float()
        validation_bce = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                early_stop_logits.float(), early_stop_labels
            )
        )
        record = {
            "epoch": float(epoch + 1),
            "train_loss": sum(losses) / max(1, len(losses)),
            "early_stop_bce": validation_bce,
        }
        history.append(record)
        print(json.dumps({"q_warmup": record}), flush=True)
        if validation_bce + 1e-5 < best_bce:
            best_bce = validation_bce
            best_state = {
                name: value.detach().cpu().clone() for name, value in q_head.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    if best_state is None:
        raise RuntimeError("Q warmup failed to produce a checkpoint")
    q_head.load_state_dict(best_state)
    margin_select_logits = _score_q_groups(
        q_head,
        terminal[margin_select_indices],
        prompt[margin_select_indices],
        group_batch=batch_groups,
        device=context.device,
    )
    margin_selection = calibrate_q_margin(
        margin_select_logits,
        correctness[margin_select_indices].float(),
        margins=tuple(float(value) for value in config["q_warmup"]["calibration_margins"]),
        bootstrap_samples=int(config["q_warmup"]["bootstrap_samples"]),
        seed=int(config["seed"]) + 1_337,
        max_degradation=float(config["q_warmup"]["maximum_allowed_degradation"]),
        min_prompts=minimum,
        min_switches=1,
    )
    safety_logits = _score_q_groups(
        q_head,
        terminal[safety_test_indices],
        prompt[safety_test_indices],
        group_batch=batch_groups,
        device=context.device,
    )
    safety_test = calibrate_q_margin(
        safety_logits,
        correctness[safety_test_indices].float(),
        # The untouched safety split evaluates exactly one preselected margin;
        # it never searches over margins a second time.
        margins=(float(margin_selection.margin),),
        bootstrap_samples=int(config["q_warmup"]["bootstrap_samples"]),
        seed=int(config["seed"]) + 1_339,
        max_degradation=float(config["q_warmup"]["maximum_allowed_degradation"]),
        min_prompts=minimum,
        min_switches=1,
    )
    from dataclasses import asdict

    q_ready = bool(margin_selection.ready and safety_test.ready)
    combined_reason = (
        "margin selection and untouched safety gate passed"
        if q_ready
        else f"selection={margin_selection.reason}; safety={safety_test.reason}"
    )
    split_counts = {name: len(value) for name, value in indices.items()}
    split_task_counts = {
        name: {
            task: sum(groups[int(index)]["task_type"] == task for index in value)
            for task in ("math", "code")
        }
        for name, value in indices.items()
    }

    _, checkpoint = _q_state_paths(config)
    _atomic_safetensors_save(
        {name: value.detach().cpu().contiguous() for name, value in q_head.state_dict().items()},
        checkpoint,
    )
    initial_adapter = _read_initial_adapter_metadata(config)
    data_fingerprints = _validated_data_fingerprints(config)
    calibration_path = _project_path(config["paths"]["q_calibration"])
    _atomic_json(
        calibration_path,
        {
            **asdict(safety_test),
            "ready": q_ready,
            "q_ready": q_ready,
            "reason": combined_reason,
            "margin": float(margin_selection.margin),
            "margin_selection": asdict(margin_selection),
            "safety_test": asdict(safety_test),
            "positive_rate_train": positive_rate,
            "train_prompt_groups": len(train_indices),
            "split_percentages": dict(split_percentages),
            "split_algorithm": Q_SPLIT_ALGORITHM,
            "split_prompt_groups": split_counts,
            "split_task_prompt_groups": split_task_counts,
            "candidate_examples": int(correctness.numel()),
            "best_early_stop_bce": best_bce,
            "history": history,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "initial_adapter_sha256": initial_adapter["sha256"],
            "data_fingerprints": data_fingerprints,
        },
    )
    print(
        json.dumps(
            {
                "q_checkpoint": str(checkpoint),
                "q_ready": q_ready,
                "margin": margin_selection.margin,
                "reason": combined_reason,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def stage_nccl_probe(config: Mapping[str, Any]) -> None:
    context = _distributed_context(config, require_distributed=True)
    try:
        value = torch.tensor(float(context.rank + 1), device=context.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = context.world_size * (context.world_size + 1) / 2
        if float(value) != expected:
            raise RuntimeError(f"NCCL all-reduce produced {float(value)}, expected {expected}")
        names: list[str | None] = [None] * context.world_size
        dist.all_gather_object(names, torch.cuda.get_device_name(context.device))
        if context.is_main:
            print(json.dumps({"nccl": "ok", "world_size": context.world_size, "gpus": names}))
    finally:
        _destroy_process_group()


def stage_shape_smoke(config: Mapping[str, Any]) -> None:
    """One-GPU real-checkpoint dimensional, zero-anchor, replay, and gradient smoke."""

    context = _distributed_context(config, require_distributed=False)
    if context.world_size != 1:
        raise RuntimeError("shape_smoke must be launched on exactly one GPU")
    _seed_everything(int(config["seed"]), context)
    policy, tokenizer = load_policy(config, context.device, load_warm_q=False)
    from .prompting import build_prefixlm_batch

    prefix = build_prefixlm_batch(
        tokenizer,
        ["What is 17 times 23?"],
        condition=str(config["prompting"]["condition"]),
        response_prefix=str(config["prompting"]["response_prefix"]),
        device=context.device,
    )
    with torch.no_grad():
        clean = policy(
            input_ids=prefix.input_ids,
            attention_mask=prefix.attention_mask,
            token_type_ids=prefix.token_type_ids,
            position_ids=prefix.position_ids,
            particle_z=None,
            particle_mask=None,
            prompt_mask=prefix.prompt_mask,
            terminal_mask=prefix.particle_mask,
            use_cache=False,
            return_q=False,
        )
        zero_z = torch.zeros(
            1,
            int(policy.adapter.latent_size),
            device=context.device,
            dtype=(
                torch.float32
                if _particle_mode(config) == "gaussian"
                else torch.bfloat16
            ),
        )
        zero = policy(
            input_ids=prefix.input_ids,
            attention_mask=prefix.attention_mask,
            token_type_ids=prefix.token_type_ids,
            position_ids=prefix.position_ids,
            particle_z=zero_z,
            particle_mask=prefix.particle_mask,
            prompt_summary=clean.prompt_summary,
            terminal_mask=prefix.particle_mask,
            use_cache=False,
            return_q=False,
        )
        if not torch.equal(clean.logits, zero.logits):
            raise RuntimeError("z=0 changed HRM logits; exact anchor invariant failed")
        if zero.injection_delta is None or bool((zero.injection_delta != 0).any()):
            raise RuntimeError("z=0 did not produce an exact-zero H intervention")

        explorer_z = torch.randn_like(zero_z)
        explorer_z /= explorer_z.float().square().mean(dim=-1, keepdim=True).sqrt().to(
            explorer_z.dtype
        )
        explorer = policy(
            input_ids=prefix.input_ids,
            attention_mask=prefix.attention_mask,
            token_type_ids=prefix.token_type_ids,
            position_ids=prefix.position_ids,
            particle_z=explorer_z,
            particle_mask=prefix.particle_mask,
            prompt_summary=clean.prompt_summary,
            terminal_mask=prefix.particle_mask,
            use_cache=False,
            return_q=False,
        )
        delta = explorer.injection_delta
        if delta is None or not bool((delta[prefix.particle_mask] != 0).any()):
            raise RuntimeError("explorer produced no H intervention")
        if bool((delta[~prefix.particle_mask] != 0).any()):
            raise RuntimeError("particle changed a bidirectional prompt position")
        relative = explorer.relative_rms.float()[prefix.particle_mask]
        maximum_relative_rms = (
            float(config["particles"]["gaussian"]["default_relative_rms"])
            if _particle_mode(config) == "gaussian"
            else float(config["adapter"]["max_relative_rms"])
        )
        if float(relative.max()) > maximum_relative_rms + 2e-3:
            raise RuntimeError("particle exceeded its relative RMS cap")

    engine = _engine(
        policy,
        tokenizer,
        config,
        max_new_tokens=8,
        seed=int(config["seed"]),
        compute_reference_logprobs=True,
    )
    rollout = engine.generate([RolloutExample("What is 17 times 23?", "391", "dummy")])
    replay = score_actions(policy, rollout)
    valid = rollout.action_mask
    if not torch.allclose(
        replay.logprobs[valid], rollout.old_logprobs[valid], atol=2e-4, rtol=2e-4
    ):
        raise RuntimeError("fresh PPO replay ratio is not one")
    rollout.rewards.copy_(torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=context.device))
    rollout.q_labels = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=context.device)
    module = RLTrainModule(policy, config).to(context.device)
    result = module(rollout)
    result["loss"].backward()
    trainable = _trainable_parameters(module)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    if float(grad_norm) <= 0:
        raise RuntimeError("real HRM smoke produced no adapter/Q gradient")
    if any(parameter.grad is not None for parameter in policy.base_model.parameters()):
        raise RuntimeError("frozen HRM backbone accumulated gradients")
    with torch.no_grad():
        init_logits = policy.q_head(
            torch.randn(4, 1536, device=context.device, dtype=torch.bfloat16),
            torch.randn(4, 1536, device=context.device, dtype=torch.bfloat16),
        )
    if not torch.equal(init_logits, init_logits[:1].expand_as(init_logits)):
        raise RuntimeError("constant-prior Q initialization did not tie all candidates")
    print(
        json.dumps(
            {
                "shape_smoke": "ok",
                "response_texts": rollout.response_texts,
                "terminal_shape": list(rollout.terminal_states.shape),
                "prompt_shape": list(rollout.prompt_summary.shape),
                "q_shape": list(_q_logits(policy, rollout).shape),
                "first_replay_mean_ratio": float(result["mean_ratio"]),
                "grad_norm": float(grad_norm),
                "max_vram_gib": torch.cuda.max_memory_allocated(context.device) / 2**30,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _ddp(module: RLTrainModule, context: DistributedContext) -> DDP:
    return DDP(
        module,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def stage_ddp_smoke(config: Mapping[str, Any]) -> None:
    context = _distributed_context(config, require_distributed=True)
    try:
        _seed_everything(int(config["seed"]) + 101, context)
        policy, tokenizer = load_policy(config, context.device, load_warm_q=False)
        module = RLTrainModule(policy, config).to(context.device)
        ddp = _ddp(module, context)
        parameters = _trainable_parameters(module)
        optimizer = BF16MasterAdamW(parameters, lr=1e-4, weight_decay=0.0)
        engine = _engine(
            policy,
            tokenizer,
            config,
            max_new_tokens=8,
            seed=int(config["seed"]) + context.rank * 100_003,
            compute_reference_logprobs=True,
        )
        example = RolloutExample(
            f"Rank {context.rank}: what is 2 plus 2?", "4", f"ddp-{context.rank}"
        )
        rollout = engine.generate([example])
        rollout.rewards.copy_(torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=context.device))
        rollout.q_labels = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=context.device)
        optimizer.zero_grad(set_to_none=True)
        output = ddp(rollout)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        checksum = _parameter_checksum(module.policy.adapter) + _parameter_checksum(
            module.policy.q_head
        )
        gathered = [torch.zeros_like(checksum) for _ in range(context.world_size)]
        dist.all_gather(gathered, checksum)
        if not all(torch.allclose(gathered[0], value, atol=1e-5, rtol=1e-6) for value in gathered):
            raise RuntimeError(f"DDP trainable weights diverged: {[float(x) for x in gathered]}")
        if context.is_main:
            print(
                json.dumps(
                    {
                        "ddp_smoke": "ok",
                        "world_size": context.world_size,
                        "checksum": float(checksum),
                        "mean_ratio": float(output["mean_ratio"]),
                    }
                ),
                flush=True,
            )
    finally:
        _destroy_process_group()


def _is_cuda_oom(error: RuntimeError) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(error, oom_type) or "out of memory" in str(error).lower()


def _pad_rollout_for_memory_probe(
    rollout: ParticleRollout, *, target_actions: int, filler_token_id: int
) -> ParticleRollout:
    """Extend an early-EOS rollout so the probe covers the configured token cap."""

    current_actions = int(rollout.action_ids.shape[-1])
    if target_actions < current_actions:
        raise ValueError("target_actions cannot be shorter than the observed rollout")
    extra = target_actions - current_actions
    if extra == 0:
        rollout.reference_logprobs = None
        return rollout
    b, k, sequence = rollout.model_input_ids.shape
    device = rollout.model_input_ids.device
    ids = torch.full(
        (b, k, extra),
        int(filler_token_id),
        device=device,
        dtype=rollout.model_input_ids.dtype,
    )
    truth = torch.ones((b, k, extra), device=device, dtype=torch.bool)
    rollout.model_input_ids = torch.cat((rollout.model_input_ids, ids), dim=-1)
    rollout.attention_mask = torch.cat((rollout.attention_mask, truth), dim=-1)
    rollout.particle_mask = torch.cat((rollout.particle_mask, truth), dim=-1)
    if rollout.token_type_ids is not None:
        rollout.token_type_ids = torch.cat(
            (rollout.token_type_ids, torch.zeros_like(ids)), dim=-1
        )
    increments = torch.arange(1, extra + 1, device=device).view(1, 1, -1)
    rollout.position_ids = torch.cat(
        (rollout.position_ids, rollout.position_ids[..., -1:] + increments), dim=-1
    )
    rollout.action_ids = torch.cat((rollout.action_ids, ids), dim=-1)
    rollout.generated_mask = torch.cat((rollout.generated_mask, truth), dim=-1)
    rollout.action_mask = torch.cat((rollout.action_mask, truth), dim=-1)
    new_action_positions = torch.arange(
        sequence - 1,
        sequence + extra - 1,
        device=device,
        dtype=rollout.action_positions.dtype,
    ).view(1, 1, -1).expand(b, k, -1)
    rollout.action_positions = torch.cat(
        (rollout.action_positions, new_action_positions), dim=-1
    )
    rollout.old_logprobs = torch.cat(
        (
            rollout.old_logprobs,
            torch.zeros(
                (b, k, extra),
                device=rollout.old_logprobs.device,
                dtype=rollout.old_logprobs.dtype,
            ),
        ),
        dim=-1,
    )
    rollout.reference_logprobs = None
    return rollout


def _measure_training_peak(
    *,
    policy: ParticleHrmForCausalLM,
    tokenizer: Any,
    module: RLTrainModule,
    config: Mapping[str, Any],
    context: DistributedContext,
    example: RolloutExample,
    prompt_batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Measure the exact rollout/reference/replay/backward peak on one rank."""

    if prompt_batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("prompt_batch_size and max_new_tokens must be positive")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(context.device)
    module.zero_grad(set_to_none=True)
    baseline_allocated = int(torch.cuda.memory_allocated(context.device))
    baseline_reserved = int(torch.cuda.memory_reserved(context.device))
    free_before, physical_total = torch.cuda.mem_get_info(context.device)
    usable_capacity = int(free_before) + baseline_reserved
    torch.cuda.reset_peak_memory_stats(context.device)
    engine = _engine(
        policy,
        tokenizer,
        config,
        max_new_tokens=max_new_tokens,
        seed=int(config["seed"]) + context.rank * 100_003 + 211,
        compute_reference_logprobs=False,
    )
    examples = [
        RolloutExample(
            example.prompt,
            example.answer,
            f"{example.example_id}-vram-{index}",
            dict(example.metadata or {}),
        )
        for index in range(prompt_batch_size)
    ]
    rollout: ParticleRollout | None = None
    output: dict[str, Tensor] | None = None
    error_message: str | None = None
    response_tokens = 0
    try:
        rollout = engine.generate(examples)
        rollout = _pad_rollout_for_memory_probe(
            rollout,
            target_actions=max_new_tokens,
            filler_token_id=int(tokenizer.eos_token_id),
        )
        with torch.no_grad():
            clean_scored = score_actions(
                policy, rollout, particle_z=torch.zeros_like(rollout.particle_z)
            )
        rollout.reference_logprobs = clean_scored.raw_logprobs.detach()
        del clean_scored
        rollout.rewards.zero_()
        rollout.rewards[:, 1] = 1.0
        rollout.q_labels = torch.zeros_like(rollout.rewards)
        rollout.q_labels[:, 1] = 1.0
        output = module(rollout)
        output["loss"].backward()
        torch.cuda.synchronize(context.device)
        response_tokens = int(rollout.action_ids.shape[-1])
        success = True
    except RuntimeError as exc:
        if not _is_cuda_oom(exc):
            raise
        success = False
        error_message = str(exc).splitlines()[0][:500]

    peak_allocated = int(torch.cuda.max_memory_allocated(context.device))
    peak_reserved = int(torch.cuda.max_memory_reserved(context.device))
    module.zero_grad(set_to_none=True)
    del output, rollout, engine, examples
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "success": success,
        "prompt_batch_size": prompt_batch_size,
        "max_new_tokens": max_new_tokens,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "free_before_bytes": int(free_before),
        "physical_total_bytes": int(physical_total),
        "usable_capacity_bytes": usable_capacity,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": peak_allocated / GIB,
        "peak_reserved_gib": peak_reserved / GIB,
        "response_tokens_observed": response_tokens,
        "oom": error_message,
    }


def stage_vram_plan(config: Mapping[str, Any], *, force: bool = False) -> None:
    """Empirically select a shared batch plan at 75% target / 80% hard cap."""

    context = _distributed_context(config, require_distributed=True)
    try:
        path = _memory_plan_path(config)
        if path.is_file() and not force:
            plan = _load_memory_plan(config, context)
            if context.is_main:
                print(json.dumps({"vram_plan": "reused", "plan": plan}, sort_keys=True))
            return
        run_directory = _project_path(config["paths"]["run_directory"])
        guarded = [
            *(
                [_noise_scale_path(config)]
                if _particle_mode(config) == "gaussian"
                else []
            ),
            _project_path(config["paths"]["q_checkpoint"]),
            _project_path(config["paths"]["q_state_directory"])
            / "collection-complete.json",
            run_directory / "pilot" / "last.json",
            run_directory / "train" / "last.json",
        ]
        if any(artifact.exists() for artifact in guarded):
            raise RuntimeError(
                "refusing to create or replace the VRAM plan after Q/training "
                "artifacts exist"
            )

        _seed_everything(int(config["seed"]) + 181, context)
        _validated_data_fingerprints(config)
        data_directory = _project_path(config["paths"]["data_directory"])
        q_records = _load_records(data_directory / "q_warm.jsonl")
        rl_records = _load_records(data_directory / "rl_train.jsonl")
        eval_records = _load_records(data_directory / "eval_math.jsonl")
        policy, tokenizer = load_policy(config, context.device, load_warm_q=False)
        module = RLTrainModule(policy, config).to(context.device)

        from .prompting import format_hrm_prompt

        generation = config["generation"]
        probe_workloads: list[tuple[RolloutExample, int, str]] = []
        for record in q_records:
            # Q prompts are generated once for warmup and again with the final
            # adapter. The latter uses task-specific evaluation lengths.
            final_cap = int(
                generation[
                    "code_eval_max_new_tokens"
                    if record["task_type"] == "code"
                    else "math_eval_max_new_tokens"
                ]
            )
            probe_workloads.append(
                (
                    _rollout_example(record, config["prompting"]),
                    max(int(generation["q_collect_max_new_tokens"]), final_cap),
                    "q_warm_and_final_q",
                )
            )
        probe_workloads.extend(
            (
                _rollout_example(record, config["prompting"]),
                int(generation["train_max_new_tokens"]),
                "rl_train",
            )
            for record in rl_records
        )
        probe_workloads.extend(
            (
                _rollout_example(record, config["prompting"]),
                int(generation["math_eval_max_new_tokens"]),
                "eval_math",
            )
            for record in eval_records
        )
        mbpp_prompts, mbpp_version = load_evalplus_mbpp_prompts()
        if mbpp_version != str(config["evaluation"]["evalplus_version"]):
            raise RuntimeError(
                f"EvalPlus version mismatch during VRAM planning: {mbpp_version}"
            )
        code_suffix = str(config["prompting"]["code_suffix"])
        probe_workloads.extend(
            (
                RolloutExample(
                    prompt.rstrip() + code_suffix,
                    "",
                    f"mbpp-memory-{index}",
                    {"task_type": "code", "source": "mbpp_plus"},
                ),
                int(generation["code_eval_max_new_tokens"]),
                "mbpp_plus",
            )
            for index, prompt in enumerate(mbpp_prompts)
        )

        def encoded_length(example: RolloutExample) -> int:
            formatted = format_hrm_prompt(
                example.prompt, str(config["prompting"]["condition"])
            )
            return len(tokenizer.encode(formatted, add_special_tokens=False))

        response_prefix_tokens = len(
            tokenizer.encode(
                str(config["prompting"]["response_prefix"]),
                add_special_tokens=False,
            )
        )
        probe_example, probe_max_new_tokens, probe_stage = max(
            probe_workloads,
            key=lambda item: encoded_length(item[0])
            + response_prefix_tokens
            + item[1],
        )
        probe_prompt_tokens = encoded_length(probe_example)
        probe_total_tokens = (
            probe_prompt_tokens + response_prefix_tokens + probe_max_new_tokens
        )
        maximum_context = int(config["model"]["expected_max_position_embeddings"])
        if probe_total_tokens > maximum_context:
            raise RuntimeError(
                "longest prepared/MBPP prompt plus the largest generation cap requires "
                f"{probe_total_tokens} tokens, exceeding context {maximum_context}"
            )
        batch_one = _measure_training_peak(
            policy=policy,
            tokenizer=tokenizer,
            module=module,
            config=config,
            context=context,
            example=probe_example,
            prompt_batch_size=1,
            max_new_tokens=probe_max_new_tokens,
        )
        if not batch_one["success"]:
            raise RuntimeError(
                "prompt batch size 1 OOMed during the real V1 memory probe; "
                "shorten generation lengths or use more VRAM"
            )

        memory = config["memory"]
        target_per_rank = int(memory["target_prompt_groups_per_rank_update"])
        candidates = valid_batch_candidates(
            memory["candidate_prompt_batch_sizes"],
            target_prompts_per_rank_update=target_per_rank,
        )
        physical_total_bytes = int(
            torch.cuda.get_device_properties(context.device).total_memory
        )
        usable_capacity_bytes = int(batch_one["usable_capacity_bytes"])
        try:
            estimate = estimate_batch_from_peak(
                total_bytes=usable_capacity_bytes,
                baseline_bytes=int(batch_one["baseline_allocated_bytes"]),
                batch_one_peak_bytes=int(batch_one["peak_allocated_bytes"]),
                candidates=candidates,
                target_prompts_per_rank_update=target_per_rank,
                target_fraction=float(memory["target_fraction"]),
                scaling_safety_factor=float(memory["scaling_safety_factor"]),
            )
            local_proposal = estimate.prompt_batch_size
            estimate_payload: dict[str, Any] = {
                "prompt_batch_size": estimate.prompt_batch_size,
                "gradient_accumulation_steps": estimate.gradient_accumulation_steps,
                "estimated_peak_bytes": estimate.estimated_peak_bytes,
                "estimated_fraction": estimate.estimated_fraction,
            }
        except RuntimeError:
            # Batch one may land slightly above the 75% target while remaining
            # under the separately enforced 80% hard cap.
            local_proposal = 1
            estimate_payload = {"prompt_batch_size": 1, "fallback": "batch-one"}

        proposal_tensor = torch.tensor(
            local_proposal, device=context.device, dtype=torch.int64
        )
        dist.all_reduce(proposal_tensor, op=dist.ReduceOp.MIN)
        shared_proposal = int(proposal_tensor)
        hard_limit = float(memory["hard_limit_fraction"])
        minimum_free_bytes = int(float(memory["minimum_free_gib"]) * GIB)
        attempts: dict[str, dict[str, Any]] = {"1": batch_one}
        selected = 0
        selected_max_fraction = 1.0
        selected_minimum_free_bytes = 0

        def validate_candidate(candidate: int) -> tuple[bool, float, int]:
            key = str(candidate)
            if key not in attempts:
                attempts[key] = _measure_training_peak(
                    policy=policy,
                    tokenizer=tokenizer,
                    module=module,
                    config=config,
                    context=context,
                    example=probe_example,
                    prompt_batch_size=candidate,
                    max_new_tokens=probe_max_new_tokens,
                )
            measurement = attempts[key]
            local_success = 1 if measurement["success"] else 0
            success_tensor = torch.tensor(
                local_success, device=context.device, dtype=torch.int64
            )
            dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
            local_fraction = (
                float(measurement["peak_reserved_bytes"]) / usable_capacity_bytes
                if local_success
                else 1.0
            )
            fraction_tensor = torch.tensor(
                local_fraction, device=context.device, dtype=torch.float64
            )
            dist.all_reduce(fraction_tensor, op=dist.ReduceOp.MAX)
            local_free = (
                max(
                    0,
                    usable_capacity_bytes - int(measurement["peak_reserved_bytes"]),
                )
                if local_success
                else 0
            )
            free_tensor = torch.tensor(
                local_free, device=context.device, dtype=torch.int64
            )
            dist.all_reduce(free_tensor, op=dist.ReduceOp.MIN)
            safe = (
                int(success_tensor) == 1
                and float(fraction_tensor) <= hard_limit
                and int(free_tensor) >= minimum_free_bytes
            )
            return safe, float(fraction_tensor), int(free_tensor)

        for candidate in reversed(
            [value for value in candidates if value <= shared_proposal]
        ):
            safe, peak_fraction, free_bytes = validate_candidate(candidate)
            if safe:
                selected = candidate
                selected_max_fraction = peak_fraction
                selected_minimum_free_bytes = free_bytes
                break
        if selected <= 0:
            raise RuntimeError(
                "no candidate batch fit the configured 80% VRAM hard cap and free-memory reserve"
            )

        # The conservative linear estimate can under-fill accelerators with a
        # nonlinear memory curve. If the measured result is below the lower
        # edge of the requested 70-80% band, validate progressively larger
        # candidates and retain the last safe one.
        lower_target = max(0.70, float(memory["target_fraction"]) - 0.05)
        if selected_max_fraction < lower_target:
            for candidate in [value for value in candidates if value > selected]:
                safe, peak_fraction, free_bytes = validate_candidate(candidate)
                if not safe:
                    break
                selected = candidate
                selected_max_fraction = peak_fraction
                selected_minimum_free_bytes = free_bytes
                if selected_max_fraction >= lower_target:
                    break

        local_report = {
            "rank": context.rank,
            "device_name": torch.cuda.get_device_name(context.device),
            "physical_total_bytes": physical_total_bytes,
            "physical_total_gib": physical_total_bytes / GIB,
            "usable_capacity_bytes": usable_capacity_bytes,
            "usable_capacity_gib": usable_capacity_bytes / GIB,
            "probe_record_id": str(probe_example.example_id),
            "probe_stage": probe_stage,
            "probe_prompt_tokens": probe_prompt_tokens,
            "probe_max_new_tokens": probe_max_new_tokens,
            "probe_total_tokens": probe_total_tokens,
            "local_estimate": estimate_payload,
            "attempts": attempts,
        }
        rank_reports: list[dict[str, Any] | None] = [None] * context.world_size
        dist.all_gather_object(rank_reports, local_report)
        if any(report is None for report in rank_reports):
            raise RuntimeError("failed to gather per-rank VRAM probe reports")
        reports = [report for report in rank_reports if report is not None]
        q_batch = min(
            int(memory["maximum_q_batch_prompt_groups"]),
            max(16, 16 * selected),
        )
        plan = {
            "format": MEMORY_PLAN_FORMAT,
            "version": MEMORY_PLAN_VERSION,
            "identity": _memory_plan_identity(config),
            "selection_basis": (
                "real longest-prompt cached rollout padded to the configured action cap + "
                "clean-reference replay + particle replay/backward; minimum safe result "
                "across ranks"
            ),
            "target_fraction": float(memory["target_fraction"]),
            "lower_target_fraction": lower_target,
            "hard_limit_fraction": hard_limit,
            "minimum_free_gib": float(memory["minimum_free_gib"]),
            "rl_prompt_micro_batch_size": selected,
            "rl_gradient_accumulation_steps": target_per_rank // selected,
            "rollout_prompt_batch_size": selected,
            "q_batch_prompt_groups": q_batch,
            "target_prompt_groups_per_rank_update": target_per_rank,
            "effective_global_prompt_groups_per_update": (
                target_per_rank * context.world_size
            ),
            "measured_peak_fraction_max": selected_max_fraction,
            "measured_minimum_free_gib": selected_minimum_free_bytes / GIB,
            "minimum_total_bytes": min(
                int(report["physical_total_bytes"]) for report in reports
            ),
            "minimum_usable_capacity_bytes": min(
                int(report["usable_capacity_bytes"]) for report in reports
            ),
            "rank_reports": reports,
        }
        if context.is_main:
            _atomic_json(path, plan)
            print(json.dumps({"vram_plan": "created", "plan": plan}, sort_keys=True))
        _barrier(context)
        _load_memory_plan(config, context)
    finally:
        _destroy_process_group()


def _pool_item(pool: Sequence[dict[str, Any]], index: int, *, seed: int) -> dict[str, Any]:
    if not pool:
        raise ValueError("cannot sample an empty task pool")
    epoch, offset = divmod(index, len(pool))
    order = list(range(len(pool)))
    random.Random(seed + epoch * 1_000_003).shuffle(order)
    return pool[order[offset]]


def _build_schedule(
    records: Sequence[dict[str, Any]],
    *,
    total_slots: int,
    code_fraction: float,
    code_enabled: bool,
    seed: int,
) -> list[dict[str, Any]]:
    if total_slots <= 0:
        raise ValueError("total_slots must be positive")
    if not 0.0 <= code_fraction < 1.0:
        raise ValueError("code_fraction must lie in [0,1)")
    math_pool = [record for record in records if record["task_type"] == "math"]
    code_pool = [record for record in records if record["task_type"] == "code"]
    if not math_pool:
        raise ValueError("RL pool contains no math records")
    if code_enabled and code_fraction > 0 and not code_pool:
        raise ValueError("code RL was enabled but the pool contains no code records")
    math_index = 0
    code_index = 0
    schedule: list[dict[str, Any]] = []
    for slot in range(total_slots):
        # The floor-difference construction produces the requested long-run
        # fraction without relying on rank-local RNG or a fragile random mix.
        wants_code = code_enabled and (
            math.floor((slot + 1) * code_fraction) > math.floor(slot * code_fraction)
        )
        if wants_code:
            record = _pool_item(code_pool, code_index, seed=seed + 23)
            code_index += 1
        else:
            record = _pool_item(math_pool, math_index, seed=seed + 11)
            math_index += 1
        schedule.append(record)
    return schedule


def _rank_microbatch_records(
    schedule: Sequence[dict[str, Any]],
    *,
    step: int,
    micro: int,
    accumulation: int,
    prompt_micro_batch_size: int,
    rank: int,
    world_size: int,
) -> Sequence[dict[str, Any]]:
    """Return one rank's disjoint slice for a synchronized optimizer update."""

    values = (step, micro, accumulation, prompt_micro_batch_size, rank, world_size)
    if any(not isinstance(value, int) for value in values):
        raise TypeError("schedule indices and batch dimensions must be integers")
    if step < 0 or micro < 0 or accumulation <= 0 or prompt_micro_batch_size <= 0:
        raise ValueError("schedule step/micro/batch values are invalid")
    if not 0 <= micro < accumulation or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("schedule micro or rank is out of range")
    prompts_per_micro = prompt_micro_batch_size * world_size
    global_slot = step * accumulation * prompts_per_micro + micro * prompts_per_micro
    rank_start = global_slot + rank * prompt_micro_batch_size
    result = schedule[rank_start : rank_start + prompt_micro_batch_size]
    if len(result) != prompt_micro_batch_size:
        raise RuntimeError("training schedule ended inside a rank microbatch")
    return result


def _code_hash() -> str:
    package_directory = Path(__file__).parent
    files = sorted(package_directory.glob("*.py"))
    files.extend(
        sorted(
            path
            for path in (PROJECT_ROOT / "scripts").glob("v1_*.py")
            if path.is_file()
        )
    )
    return _canonical_hash(
        {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in files
        }
    )


def _provenance(
    config: Mapping[str, Any], context: DistributedContext, *, stage: str
) -> dict[str, Any]:
    if stage not in {"pilot", "train"}:
        raise ValueError("checkpoint stage must be 'pilot' or 'train'")
    calibration = _project_path(config["paths"]["q_calibration"])
    memory_plan = _memory_plan_path(config)
    _load_memory_plan(config, context)
    initial_adapter = _read_initial_adapter_metadata(config)
    return {
        "checkpoint_stage": stage,
        "world_size": context.world_size,
        "model_id": config["model"]["pretrained_model_name_or_path"],
        "model_revision": config["model"]["revision"],
        "config_hash": _canonical_hash(config),
        "prepared_data": _validated_data_fingerprints(config),
        "initial_adapter_sha256": initial_adapter["sha256"],
        "q_calibration_sha256": sha256_file(calibration),
        "memory_plan_sha256": sha256_file(memory_plan),
        "code_hash": _code_hash(),
        "package_versions": package_versions(),
    }


def _local_rng_bundle(engine: ParticleRolloutEngine, context: DistributedContext) -> dict[str, Any]:
    return {
        "rank": context.rank,
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": torch.cuda.get_rng_state(context.device).cpu(),
        "rollout": {key: value.cpu() for key, value in engine.rng_state_dict().items()},
    }


def _gather_rng_bundles(
    engine: ParticleRolloutEngine, context: DistributedContext
) -> list[dict[str, Any]]:
    local = _local_rng_bundle(engine, context)
    if context.world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("failed to gather per-rank RNG state")
    return [item for item in gathered if item is not None]


def _restore_rng_bundle(
    bundles: Sequence[Mapping[str, Any]],
    engine: ParticleRolloutEngine,
    context: DistributedContext,
) -> None:
    if len(bundles) != context.world_size:
        raise RuntimeError("checkpoint RNG world size changed")
    bundle = bundles[context.rank]
    if int(bundle["rank"]) != context.rank:
        raise RuntimeError("checkpoint RNG bundles are not rank ordered")
    random.setstate(bundle["python"])
    torch.set_rng_state(bundle["torch_cpu"].cpu())
    torch.cuda.set_rng_state(bundle["torch_cuda"].cpu(), context.device)
    engine.load_rng_state_dict(bundle["rollout"])


def _save_training_checkpoint(
    directory: Path,
    *,
    step: int,
    module: RLTrainModule,
    optimizer: BF16MasterAdamW,
    engine: ParticleRolloutEngine,
    context: DistributedContext,
    provenance: Mapping[str, Any],
    stage: str,
    stopped_for_budget: bool,
) -> Path | None:
    if stage not in {"pilot", "train"}:
        raise ValueError("checkpoint stage must be 'pilot' or 'train'")
    rng = _gather_rng_bundles(engine, context)
    _barrier(context)
    checkpoint: Path | None = None
    if context.is_main:
        checkpoint = directory / f"checkpoint-{step:06d}.pt"
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "stage": stage,
            "step": int(step),
            "adapter": {
                key: value.detach().cpu()
                for key, value in module.policy.adapter.state_dict().items()
            },
            "q_head": {
                key: value.detach().cpu()
                for key, value in module.policy.q_head.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "rng_by_rank": rng,
            "provenance": dict(provenance),
            "stopped_for_budget": bool(stopped_for_budget),
        }
        atomic_torch_save(payload, checkpoint)
        digest = sha256_file(checkpoint)
        checkpoint.with_suffix(checkpoint.suffix + ".sha256").write_text(digest + "\n")
        _atomic_json(
            directory / "last.json",
            {
                "checkpoint": checkpoint.name,
                "sha256": digest,
                "step": step,
                "stage": stage,
            },
        )
    _barrier(context)
    return checkpoint


def _validate_checkpoint_header(payload: Mapping[str, Any], *, expected_stage: str) -> None:
    if expected_stage not in {"pilot", "train"}:
        raise ValueError("expected checkpoint stage must be 'pilot' or 'train'")
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("version") != CHECKPOINT_VERSION
    ):
        raise RuntimeError("unsupported V1 checkpoint")
    if payload.get("stage") != expected_stage:
        raise RuntimeError(
            f"checkpoint stage {payload.get('stage')!r} cannot be used as {expected_stage!r}"
        )


def _load_training_checkpoint(
    path: Path,
    *,
    module: RLTrainModule,
    optimizer: BF16MasterAdamW,
    engine: ParticleRolloutEngine,
    context: DistributedContext,
    provenance: Mapping[str, Any],
    expected_stage: str,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not checksum_path.is_file() or sha256_file(path) != checksum_path.read_text().strip():
        raise RuntimeError("training checkpoint checksum failed")
    payload = torch.load(path, map_location=context.device, weights_only=False)
    _validate_checkpoint_header(payload, expected_stage=expected_stage)
    if payload.get("provenance") != dict(provenance):
        differing = sorted(
            key
            for key in set(payload.get("provenance", {})) | set(provenance)
            if payload.get("provenance", {}).get(key) != provenance.get(key)
        )
        raise RuntimeError("resume provenance changed: " + ", ".join(differing))
    module.policy.adapter.load_state_dict(payload["adapter"], strict=True)
    module.policy.q_head.load_state_dict(payload["q_head"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng_bundle(payload["rng_by_rank"], engine, context)
    return int(payload["step"])


def _all_reduce_metrics(
    values: Mapping[str, float], context: DistributedContext
) -> dict[str, float]:
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], device=context.device, dtype=torch.float64)
    if context.world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= context.world_size
    return {key: float(value) for key, value in zip(keys, tensor.cpu())}


_STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _distributed_should_stop(context: DistributedContext, deadline: float) -> bool:
    local = int(_STOP_REQUESTED or time.time() >= deadline)
    value = torch.tensor(local, device=context.device, dtype=torch.int32)
    if context.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(int(value))


def _load_q_calibration(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _project_path(config["paths"]["q_calibration"])
    if not path.is_file():
        raise FileNotFoundError(f"Q calibration is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = _project_path(config["paths"]["q_checkpoint"])
    if payload.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("Q checkpoint does not match its calibration record")
    initial_adapter = _read_initial_adapter_metadata(config)
    if payload.get("initial_adapter_sha256") != initial_adapter["sha256"]:
        raise RuntimeError("Q calibration does not match the persisted initial adapter")
    if payload.get("data_fingerprints") != _validated_data_fingerprints(config):
        raise RuntimeError("Q calibration does not match the prepared data")
    if payload.get("split_percentages") != dict(config["q_warmup"]["split_percentages"]):
        raise RuntimeError("Q calibration split definition changed")
    if payload.get("split_algorithm") != Q_SPLIT_ALGORITHM:
        raise RuntimeError("Q calibration split algorithm changed")
    return payload


def _pilot_code_enabled(config: Mapping[str, Any], run_directory: Path) -> tuple[bool, str]:
    gate_path = run_directory / "pilot" / "code-signal.json"
    if not gate_path.is_file():
        raise FileNotFoundError(
            f"code signal gate is missing: {gate_path}; run the rl_pilot stage first"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    return bool(gate["enabled"]), str(gate["reason"])


def _calibrate_final_q_rows(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Choose and independently test one Q margin per task type."""

    if not rows:
        raise ValueError("final Q calibration rows must not be empty")
    q_config = config["q_warmup"]
    k = int(config["particles"]["count"])
    result: dict[str, Any] = {}
    from dataclasses import asdict

    for task_type, minimum_key in (
        ("math", "minimum_math_calibration_prompts"),
        ("code", "minimum_code_calibration_prompts"),
    ):
        task_rows = [row for row in rows if row.get("task_type") == task_type]
        selection_rows = [row for row in task_rows if row.get("split") == "margin_select"]
        safety_rows = [row for row in task_rows if row.get("split") == "safety_test"]
        minimum = int(q_config[minimum_key])
        if len(selection_rows) < minimum or len(safety_rows) < minimum:
            raise RuntimeError(
                f"final {task_type} Q gate needs {minimum} prompts in each calibration split; "
                f"found selection={len(selection_rows)}, safety={len(safety_rows)}"
            )

        def matrices(values: Sequence[Mapping[str, Any]]) -> tuple[Tensor, Tensor]:
            logits = torch.tensor([row["logits"] for row in values], dtype=torch.float32)
            labels = torch.tensor(
                [row["correctness"] for row in values], dtype=torch.float32
            )
            if logits.shape != (len(values), k) or labels.shape != logits.shape:
                raise RuntimeError(f"malformed final {task_type} Q calibration rows")
            return logits, labels

        selection_logits, selection_labels = matrices(selection_rows)
        selection = calibrate_q_margin(
            selection_logits,
            selection_labels,
            margins=tuple(float(value) for value in q_config["calibration_margins"]),
            bootstrap_samples=int(q_config["bootstrap_samples"]),
            seed=int(config["seed"]) + (1_741 if task_type == "math" else 1_743),
            max_degradation=float(q_config["maximum_allowed_degradation"]),
            min_prompts=minimum,
            min_switches=1,
        )
        safety_logits, safety_labels = matrices(safety_rows)
        safety = calibrate_q_margin(
            safety_logits,
            safety_labels,
            # This split certifies a frozen margin; it does not search again.
            margins=(float(selection.margin),),
            bootstrap_samples=int(q_config["bootstrap_samples"]),
            seed=int(config["seed"]) + (1_751 if task_type == "math" else 1_753),
            max_degradation=float(q_config["maximum_allowed_degradation"]),
            min_prompts=minimum,
            min_switches=1,
        )
        ready = bool(selection.ready and safety.ready)
        result[task_type] = {
            "ready": ready,
            "q_ready": ready,
            "margin": float(selection.margin),
            "reason": (
                "disjoint margin-selection and safety-test gates passed"
                if ready
                else f"selection={selection.reason}; safety={safety.reason}"
            ),
            "selection": asdict(selection),
            "safety_test": asdict(safety),
            "selection_prompts": len(selection_rows),
            "safety_prompts": len(safety_rows),
        }
    return result


def stage_rl(
    config: Mapping[str, Any],
    *,
    pilot: bool,
    resume_from: Path | None,
) -> None:
    """One fresh rollout and one clipped replay per update, synchronized by DDP."""

    if _particle_mode(config) == "gaussian":
        raise ValueError(
            "gaussian particle mode is parameter-free and Q-only; RL/PPO cannot "
            "update fixed noise. Use noise_scale_sweep, collect_q, and q_warmup."
        )
    context = _distributed_context(config, require_distributed=True)
    try:
        segment_started = time.time()
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        raw_deadline = os.environ.get("HRM_V1_DEADLINE_UNIX")
        if not pilot and raw_deadline is None:
            raise RuntimeError(
                "full RL requires a finite HRM_V1_DEADLINE_UNIX; use the notebook's "
                "time-boxed launch"
            )
        deadline = float(raw_deadline) if raw_deadline is not None else float("inf")
        if not pilot and (not math.isfinite(deadline) or deadline <= time.time()):
            raise RuntimeError("full RL deadline must be finite and in the future")
        # Pilot and full run intentionally start from the same adapter/Q
        # initialization; only rollout/data RNG streams differ.
        _seed_everything(int(config["seed"]) + 313, context)
        memory_plan = _load_memory_plan(config, context)
        _load_q_calibration(config)
        _validated_data_fingerprints(config)
        run_directory = _project_path(config["paths"]["run_directory"])
        stage_directory = run_directory / ("pilot" if pilot else "train")
        stage_directory.mkdir(parents=True, exist_ok=True)
        metrics_path = stage_directory / "metrics.jsonl"
        if pilot:
            code_enabled, gate_reason = True, "pilot measures initial code reward density"
        else:
            code_enabled, gate_reason = _pilot_code_enabled(config, run_directory)
        if context.is_main:
            print(
                json.dumps(
                    {"code_rl_enabled": code_enabled, "reason": gate_reason, "pilot": pilot}
                ),
                flush=True,
            )

        records = _load_records(
            _project_path(config["paths"]["data_directory"]) / "rl_train.jsonl"
        )
        max_updates = int(config["rl"]["pilot_updates"] if pilot else config["rl"]["max_updates"])
        accumulation = int(memory_plan["rl_gradient_accumulation_steps"])
        prompt_micro_batch_size = int(memory_plan["rl_prompt_micro_batch_size"])
        total_slots = (
            max_updates
            * accumulation
            * prompt_micro_batch_size
            * context.world_size
        )
        schedule = _build_schedule(
            records,
            total_slots=total_slots,
            code_fraction=float(config["rl"]["code_fraction"]),
            code_enabled=code_enabled,
            seed=int(config["seed"]) + (401 if pilot else 409),
        )
        router = _reward_scorers(
            config,
            need_code=any(record["task_type"] == "code" for record in schedule),
        )
        policy, tokenizer = load_policy(config, context.device, load_warm_q=True)
        _load_initial_adapter_artifact(policy.adapter, config)
        _synchronize_small_modules(policy, context)
        module = RLTrainModule(policy, config).to(context.device)
        ddp = _ddp(module, context)
        optimizer = BF16MasterAdamW(
            [
                {
                    "params": list(module.policy.adapter.parameters()),
                    "lr": float(config["rl"]["actor_learning_rate"]),
                    "weight_decay": float(config["rl"]["weight_decay"]),
                },
                {
                    "params": list(module.policy.q_head.parameters()),
                    "lr": float(config["rl"]["q_learning_rate"]),
                    "weight_decay": float(config["rl"]["weight_decay"]),
                },
            ]
        )
        engine = _engine(
            policy,
            tokenizer,
            config,
            max_new_tokens=int(config["generation"]["train_max_new_tokens"]),
            seed=int(config["seed"]) + context.rank * 100_003 + (503 if pilot else 509),
            compute_reference_logprobs=True,
        )
        checkpoint_stage = "pilot" if pilot else "train"
        provenance = _provenance(config, context, stage=checkpoint_stage)
        start_step = 0
        if resume_from is not None:
            if pilot:
                raise ValueError("pilot resume is intentionally disabled; rerun its short window")
            start_step = _load_training_checkpoint(
                resume_from,
                module=module,
                optimizer=optimizer,
                engine=engine,
                context=context,
                provenance=provenance,
                expected_stage="train",
            )
        elif (
            (stage_directory / "last.json").exists()
            or (metrics_path.exists() and metrics_path.stat().st_size > 0)
            or any(stage_directory.glob("checkpoint-*.pt"))
        ):
            raise ValueError(
                f"refusing to overwrite existing artifacts in {stage_directory}; "
                "pass --resume-from for full RL"
            )

        elapsed_offset = 0.0
        if context.is_main:
            elapsed_offset = _reconcile_metrics_log(metrics_path, start_step)
        elapsed_tensor = torch.tensor(elapsed_offset, device=context.device, dtype=torch.float64)
        if context.world_size > 1:
            dist.broadcast(elapsed_tensor, src=0)
        elapsed_offset = float(elapsed_tensor)
        code_correctness_local: list[list[float]] = []
        stopped = False
        maximum_update_seconds = 0.0
        for step in range(start_step, max_updates):
            update_started = time.time()
            optimizer.zero_grad(set_to_none=True)
            micro_metrics: list[dict[str, float]] = []
            for micro in range(accumulation):
                record_batch = _rank_microbatch_records(
                    schedule,
                    step=step,
                    micro=micro,
                    accumulation=accumulation,
                    prompt_micro_batch_size=prompt_micro_batch_size,
                    rank=context.rank,
                    world_size=context.world_size,
                )
                examples = [
                    _rollout_example(record, config["prompting"])
                    for record in record_batch
                ]
                rollout = engine.generate(examples)
                rollout = _rescore(rollout, examples, router=router)
                strict_labels = _strict_q_labels(rollout)
                for batch_index, record in enumerate(record_batch):
                    if record["task_type"] == "code":
                        code_correctness_local.append(
                            strict_labels[batch_index].cpu().tolist()
                        )
                sync_context = ddp.no_sync() if micro + 1 < accumulation else nullcontext()
                with sync_context:
                    output = ddp(rollout)
                    (output["loss"] / accumulation).backward()
                labels = strict_labels
                rewards = rollout.rewards.float()
                micro_metrics.append(
                    {
                        "loss": float(output["loss"].detach()),
                        "actor_loss": float(output["actor_loss"].detach()),
                        "q_loss": float(output["q_loss"].detach()),
                        "reference_kl": float(output["reference_kl"].detach()),
                        "mean_ratio": float(output["mean_ratio"].detach()),
                        "clip_fraction": float(output["clip_fraction"].detach()),
                        "injection_penalty": float(output["injection_penalty"].detach()),
                        "anchor_actor_reward": float(rewards[:, 0].mean()),
                        "oracle_actor_reward": float(rewards.max(dim=1).values.mean()),
                        "anchor_correctness": float(labels[:, 0].mean()),
                        "oracle_correctness": float(labels.max(dim=1).values.mean()),
                        "mixed_correctness": float(
                            ((labels.sum(dim=1) > 0) & (labels.sum(dim=1) < labels.shape[1]))
                            .float()
                            .mean()
                        ),
                    }
                )
            actor_norm = torch.nn.utils.clip_grad_norm_(
                list(module.policy.adapter.parameters()),
                float(config["rl"]["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            q_norm = torch.nn.utils.clip_grad_norm_(
                list(module.policy.q_head.parameters()),
                float(config["rl"]["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            local_average = {
                key: sum(item[key] for item in micro_metrics) / len(micro_metrics)
                for key in micro_metrics[0]
            }
            local_average["actor_grad_norm"] = float(actor_norm)
            local_average["q_grad_norm"] = float(q_norm)
            metrics = _all_reduce_metrics(local_average, context)
            metrics.update(
                {
                    "step": float(step + 1),
                    "code_rl_enabled": float(code_enabled),
                    "elapsed_seconds": float(
                        elapsed_offset + time.time() - segment_started
                    ),
                    "wall_time_unix": float(time.time()),
                    "prompt_micro_batch_size": float(prompt_micro_batch_size),
                    "gradient_accumulation_steps": float(accumulation),
                    "effective_global_prompt_groups": float(
                        prompt_micro_batch_size * accumulation * context.world_size
                    ),
                }
            )
            update_seconds = time.time() - update_started
            maximum_update_seconds = max(maximum_update_seconds, update_seconds)
            metrics["update_seconds"] = float(update_seconds)
            metrics["maximum_update_seconds_this_segment"] = float(
                maximum_update_seconds
            )
            metrics["next_update_reserve_seconds"] = float(
                maximum_update_seconds * 1.25
            )
            if step == start_step and abs(metrics["mean_ratio"] - 1.0) > 2e-3:
                raise RuntimeError(
                    f"first PPO replay ratio is {metrics['mean_ratio']:.6f}, expected one"
                )
            if context.is_main:
                _append_jsonl(metrics_path, metrics)
                if (step + 1) % int(config["runtime"]["log_every_updates"]) == 0:
                    print(json.dumps({"training": metrics}, sort_keys=True), flush=True)
            stopped = _distributed_should_stop(
                context, deadline - maximum_update_seconds * 1.25
            )
            should_checkpoint = (
                (step + 1) % int(config["runtime"]["checkpoint_every_updates"]) == 0
                or step + 1 == max_updates
                or stopped
            )
            if should_checkpoint:
                _save_training_checkpoint(
                    stage_directory,
                    step=step + 1,
                    module=module,
                    optimizer=optimizer,
                    engine=engine,
                    context=context,
                    provenance=provenance,
                    stage=checkpoint_stage,
                    stopped_for_budget=stopped,
                )
            if stopped:
                break

        if pilot:
            gathered_code: list[list[list[float]] | None] = [None] * context.world_size
            dist.all_gather_object(gathered_code, code_correctness_local)
            if context.is_main:
                flattened = [
                    group
                    for rank_groups in gathered_code
                    if rank_groups is not None
                    for group in rank_groups
                ]
                if not flattened:
                    raise RuntimeError("RL pilot scheduled no code groups")
                gate = code_signal_gate(
                    torch.tensor(flattened, dtype=torch.float32),
                    min_groups=16,
                    min_nonzero_groups=int(config["rl"]["code_min_nonzero_groups"]),
                    min_mixed_groups=2,
                    min_nonzero_fraction=float(
                        config["rl"]["code_min_nonzero_fraction"]
                    ),
                    min_mixed_fraction=float(config["rl"]["code_min_mixed_fraction"]),
                )
                from dataclasses import asdict

                _atomic_json(stage_directory / "code-signal.json", asdict(gate))
                print(json.dumps({"code_signal_gate": asdict(gate)}, sort_keys=True), flush=True)
    finally:
        _destroy_process_group()


def stage_final_q_calibration(
    config: Mapping[str, Any], *, checkpoint: Path | None
) -> None:
    """Regenerate held-out states under the final adapter and gate Q by task."""

    context = _distributed_context(config, require_distributed=True)
    try:
        _seed_everything(int(config["seed"]) + 557, context)
        memory_plan = _load_memory_plan(config, context)
        prompt_batch_size = int(memory_plan["rollout_prompt_batch_size"])
        data_fingerprints = _validated_data_fingerprints(config)
        checkpoint = _resolve_eval_checkpoint(config, checkpoint)
        policy, tokenizer = load_policy(config, context.device, load_warm_q=False)
        loaded = _load_inference_checkpoint(
            checkpoint, policy=policy, config=config, context=context
        )
        records = _load_records(
            _project_path(config["paths"]["data_directory"]) / "q_warm.jsonl"
        )
        percentages = config["q_warmup"]["split_percentages"]
        assignments = stratified_prompt_splits(
            [str(record["id"]) for record in records],
            [str(record.get("source", "")) for record in records],
            percentages,
        )
        split_by_id = {
            str(record["id"]): split for record, split in zip(records, assignments)
        }
        held_out = [
            record
            for record in records
            if split_by_id[str(record["id"])] in {"margin_select", "safety_test"}
        ]
        local_records = held_out[context.rank :: context.world_size]
        router = _reward_scorers(config, need_code=True)
        math_engine = _engine(
            policy,
            tokenizer,
            config,
            max_new_tokens=int(config["generation"]["math_eval_max_new_tokens"]),
            seed=int(config["seed"]) + context.rank * 100_003 + 563,
            compute_reference_logprobs=False,
        )
        code_engine = _engine(
            policy,
            tokenizer,
            config,
            max_new_tokens=int(config["generation"]["code_eval_max_new_tokens"]),
            seed=int(config["seed"]) + context.rank * 100_003 + 569,
            compute_reference_logprobs=False,
        )
        rows: list[dict[str, Any]] = []
        processed = 0
        # Keep math and code batches separate so each uses its configured
        # generation length while still filling the empirically safe batch.
        for task_type, engine in (("math", math_engine), ("code", code_engine)):
            task_records = [
                record for record in local_records if record["task_type"] == task_type
            ]
            for record_batch in _chunks(task_records, prompt_batch_size):
                examples = [
                    _rollout_example(record, config["prompting"])
                    for record in record_batch
                ]
                rollout = _rescore(
                    engine.generate(examples), examples, router=router
                )
                with torch.no_grad():
                    logits_batch = _q_logits(policy, rollout).float().cpu()
                labels_batch = _strict_q_labels(rollout).cpu()
                for batch_index, record in enumerate(record_batch):
                    rows.append(
                        {
                            "id": str(record["id"]),
                            "task_type": str(record["task_type"]),
                            "source": str(record.get("source", "")),
                            "split": split_by_id[str(record["id"])],
                            "logits": [
                                float(value) for value in logits_batch[batch_index]
                            ],
                            "correctness": [
                                int(value) for value in labels_batch[batch_index]
                            ],
                        }
                    )
                processed += len(record_batch)
                if context.is_main and processed % 20 < len(record_batch):
                    print(
                        json.dumps({"final_q_groups_rank0": processed}), flush=True
                    )

        run_directory = _project_path(config["paths"]["run_directory"])
        state_directory = run_directory / "final_q_calibration_states"
        shard = state_directory / f"rank-{context.rank:02d}.jsonl"
        _write_jsonl_records(shard, rows)
        _barrier(context)
        if context.is_main:
            merged: list[dict[str, Any]] = []
            shard_hashes: dict[str, str] = {}
            shard_paths = sorted(state_directory.glob("rank-*.jsonl"))
            if len(shard_paths) != context.world_size:
                raise RuntimeError(
                    f"final Q calibration found {len(shard_paths)} shards; "
                    f"expected {context.world_size}"
                )
            for path in shard_paths:
                shard_hashes[path.name] = sha256_file(path)
                with path.open("r", encoding="utf-8") as handle:
                    merged.extend(json.loads(line) for line in handle if line.strip())
            expected_ids = {str(record["id"]) for record in held_out}
            observed_ids = [str(row["id"]) for row in merged]
            if len(observed_ids) != len(expected_ids) or set(observed_ids) != expected_ids:
                raise RuntimeError("final Q calibration shards are missing/duplicating prompt groups")
            gates = _calibrate_final_q_rows(merged, config)
            destination = _project_path(config["paths"]["final_q_calibration"])
            payload = {
                "format": "hrm-particle-v1-final-q-calibration",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_step": int(loaded["step"]),
                "training_stopped_for_budget": bool(
                    loaded.get("stopped_for_budget", False)
                ),
                "checkpoint_provenance_hash": _canonical_hash(loaded["provenance"]),
                "data_fingerprints": data_fingerprints,
                "split_percentages": dict(percentages),
                "split_algorithm": Q_SPLIT_ALGORITHM,
                "prompt_groups": len(merged),
                "shard_sha256": shard_hashes,
                "gates": gates,
                "note": (
                    "Fresh candidates/states were generated with the final adapter. "
                    "Margins are selected and safety-tested on disjoint prompt groups."
                ),
            }
            _atomic_json(destination, payload)
            print(json.dumps({"final_q_calibration": payload}, sort_keys=True), flush=True)
    finally:
        _destroy_process_group()


def _resolve_eval_checkpoint(config: Mapping[str, Any], requested: Path | None) -> Path:
    if _particle_mode(config) == "gaussian":
        q_checkpoint = _project_path(config["paths"]["q_checkpoint"])
        if requested is not None and requested.resolve() != q_checkpoint:
            raise ValueError(
                "gaussian Q-only evaluation accepts only the configured q_checkpoint"
            )
        if not q_checkpoint.is_file():
            raise FileNotFoundError(
                f"warm Q checkpoint is missing: {q_checkpoint}; run q_warmup first"
            )
        return q_checkpoint
    if requested is not None:
        return requested.resolve()
    train_directory = _project_path(config["paths"]["run_directory"]) / "train"
    last_path = train_directory / "last.json"
    if not last_path.is_file():
        raise FileNotFoundError("no trained checkpoint found; pass --checkpoint explicitly")
    last = json.loads(last_path.read_text(encoding="utf-8"))
    if last.get("stage") != "train":
        raise RuntimeError("train/last.json does not identify a full-train checkpoint")
    return train_directory / str(last["checkpoint"])


def _load_inference_checkpoint(
    checkpoint: Path,
    *,
    policy: ParticleHrmForCausalLM,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> dict[str, Any]:
    if _particle_mode(config) == "gaussian":
        expected_checkpoint = _project_path(config["paths"]["q_checkpoint"])
        if checkpoint.resolve() != expected_checkpoint:
            raise ValueError(
                "gaussian Q-only inference requires the configured q_checkpoint"
            )
        selection = _load_noise_scale_selection(config)
        initial_adapter = _load_initial_adapter_artifact(policy.adapter, config)
        q_calibration = _load_q_calibration(config)
        _load_q_weights(policy.q_head, checkpoint)
        selected = float(selection["selected_relative_rms"])
        observed = float(policy.adapter.relative_rms_scale.detach().cpu())
        if not math.isclose(observed, selected, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(
                "persisted Gaussian intervention scale does not match scale selection"
            )
        return {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "stage": "gaussian-q-only",
            "step": 0,
            "stopped_for_budget": False,
            "provenance": {
                "particle_mode": "gaussian",
                "model_revision": str(config["model"]["revision"]),
                "q_checkpoint_sha256": sha256_file(checkpoint),
                "q_calibration_sha256": sha256_file(
                    _project_path(config["paths"]["q_calibration"])
                ),
                "initial_adapter_sha256": initial_adapter["sha256"],
                "noise_scale_selection_sha256": sha256_file(
                    _noise_scale_path(config)
                ),
                "selected_relative_rms": selected,
                "data_fingerprints": q_calibration["data_fingerprints"],
            },
        }
    checksum_path = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if not checksum_path.is_file() or sha256_file(checkpoint) != checksum_path.read_text().strip():
        raise RuntimeError("evaluation checkpoint checksum failed")
    payload = torch.load(checkpoint, map_location=context.device, weights_only=False)
    _validate_checkpoint_header(payload, expected_stage="train")
    expected = _provenance(config, context, stage="train")
    if payload.get("provenance") != expected:
        raise RuntimeError("evaluation checkpoint provenance does not match this run")
    policy.adapter.load_state_dict(payload["adapter"], strict=True)
    policy.q_head.load_state_dict(payload["q_head"], strict=True)
    return payload


def _load_final_q_gate(
    config: Mapping[str, Any], checkpoint: Path, *, task_type: str
) -> dict[str, Any]:
    """Load a task-specific final-state Q gate, or safely retain branch zero."""

    if task_type not in {"math", "code"}:
        raise ValueError("task_type must be math or code")
    path = _project_path(config["paths"]["final_q_calibration"])
    if not path.is_file():
        return {
            "ready": False,
            "q_ready": False,
            "margin": 0.0,
            "reason": f"final Q calibration is missing: {path}; branch zero forced",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "hrm-particle-v1-final-q-calibration":
        raise RuntimeError("unsupported final Q calibration artifact")
    if payload.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("final Q calibration was generated from a different checkpoint")
    if payload.get("data_fingerprints") != _validated_data_fingerprints(config):
        raise RuntimeError("final Q calibration was generated from different prepared data")
    if payload.get("split_percentages") != dict(config["q_warmup"]["split_percentages"]):
        raise RuntimeError("final Q calibration split definition changed")
    if payload.get("split_algorithm") != Q_SPLIT_ALGORITHM:
        raise RuntimeError("final Q calibration split algorithm changed")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or not isinstance(gates.get(task_type), Mapping):
        raise RuntimeError(f"final Q calibration has no {task_type} gate")
    gate = dict(gates[task_type])
    if not isinstance(gate.get("ready"), bool):
        raise RuntimeError("final Q gate readiness is malformed")
    margin = float(gate.get("margin", float("nan")))
    if not math.isfinite(margin) or margin < 0.0:
        raise RuntimeError("final Q gate margin is malformed")
    gate["artifact"] = str(path)
    gate["checkpoint_step"] = int(payload["checkpoint_step"])
    return gate


def _math_equivalence_majority_vote(
    texts: Sequence[str],
    *,
    parse_fn: Any | None = None,
    verify_fn: Any | None = None,
) -> tuple[int, list[str | None]]:
    """Cluster four answers by mathematical equivalence, then tie-break early."""

    if not texts:
        raise ValueError("math majority vote requires at least one response")
    if parse_fn is None or verify_fn is None:
        try:
            from math_verify import parse, verify
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("math-verify is required for majority voting") from exc
        parse_fn = parse if parse_fn is None else parse_fn
        verify_fn = verify if verify_fn is None else verify_fn
    parsed_values: list[Any | None] = []
    display: list[str | None] = []
    for text in texts:
        try:
            parsed = parse_fn(text)
        except Exception:
            parsed = None
        if not parsed:
            parsed_values.append(None)
            display.append(None)
        else:
            parsed_values.append(parsed)
            display.append(repr(parsed[0]))

    clusters: list[dict[str, Any]] = []
    for index, parsed in enumerate(parsed_values):
        if parsed is None:
            continue
        destination = None
        for cluster in clusters:
            try:
                equivalent = bool(verify_fn(cluster["representative"], parsed))
            except Exception:
                equivalent = False
            if equivalent:
                destination = cluster
                break
        if destination is None:
            clusters.append(
                {"representative": parsed, "first_index": index, "count": 1}
            )
        else:
            destination["count"] += 1
    if not clusters:
        return 0, display
    winner = max(clusters, key=lambda cluster: (cluster["count"], -cluster["first_index"]))
    return int(winner["first_index"]), display


def _write_jsonl_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _clustered_bootstrap_delta(
    system: Sequence[float],
    baseline: Sequence[float],
    clusters: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if not (len(system) == len(baseline) == len(clusters)) or not system:
        raise ValueError("clustered bootstrap inputs must be non-empty and aligned")
    grouped: dict[str, list[float]] = {}
    for system_value, baseline_value, cluster in zip(system, baseline, clusters):
        grouped.setdefault(str(cluster), []).append(float(system_value) - float(baseline_value))
    cluster_means = torch.tensor(
        [sum(values) / len(values) for values in grouped.values()], dtype=torch.float64
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(cluster_means),
        (samples, len(cluster_means)),
        generator=generator,
    )
    bootstrap = cluster_means[indices].mean(dim=1)
    return {
        "mean_delta": float(cluster_means.mean()),
        "low": float(torch.quantile(bootstrap, 0.025)),
        "high": float(torch.quantile(bootstrap, 0.975)),
        "confidence": 0.95,
        "clusters": float(len(cluster_means)),
    }


def _aggregate_math_eval(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    from .evaluate import paired_bootstrap_delta
    from dataclasses import asdict

    result: dict[str, Any] = {}
    for benchmark in sorted({str(record["benchmark"]) for record in records}):
        rows = [record for record in records if record["benchmark"] == benchmark]
        fields = (
            "anchor_correct",
            "ordinary_mean_correct",
            "ordinary_oracle_correct",
            "ordinary_majority_correct",
            "matched_zero_latent_mean_correct",
            "matched_zero_latent_oracle_correct",
            "matched_zero_latent_majority_correct",
            "particle_mean_correct",
            "particle_oracle_correct",
            "q_selected_correct",
        )
        metrics = {
            field: sum(float(row[field]) for row in rows) / len(rows) for field in fields
        }
        metrics["prompts"] = len(rows)
        if benchmark == "gsm_symbolic":
            clusters = [str(row["cluster_id"]) for row in rows]
            q_delta = _clustered_bootstrap_delta(
                [row["q_selected_correct"] for row in rows],
                [row["anchor_correct"] for row in rows],
                clusters,
                samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["seed"]) + 7,
            )
            oracle_delta = _clustered_bootstrap_delta(
                [row["particle_oracle_correct"] for row in rows],
                [row["ordinary_oracle_correct"] for row in rows],
                clusters,
                samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["seed"]) + 11,
            )
            matched_oracle_delta = _clustered_bootstrap_delta(
                [row["particle_oracle_correct"] for row in rows],
                [row["matched_zero_latent_oracle_correct"] for row in rows],
                clusters,
                samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["seed"]) + 13,
            )
            metrics["bootstrap_unit"] = "original GSM template"
        else:
            q_delta = asdict(
                paired_bootstrap_delta(
                    [row["q_selected_correct"] for row in rows],
                    [row["anchor_correct"] for row in rows],
                    num_samples=int(config["evaluation"]["bootstrap_samples"]),
                    seed=int(config["seed"]) + 7,
                )
            )
            oracle_delta = asdict(
                paired_bootstrap_delta(
                    [row["particle_oracle_correct"] for row in rows],
                    [row["ordinary_oracle_correct"] for row in rows],
                    num_samples=int(config["evaluation"]["bootstrap_samples"]),
                    seed=int(config["seed"]) + 11,
                )
            )
            matched_oracle_delta = asdict(
                paired_bootstrap_delta(
                    [row["particle_oracle_correct"] for row in rows],
                    [row["matched_zero_latent_oracle_correct"] for row in rows],
                    num_samples=int(config["evaluation"]["bootstrap_samples"]),
                    seed=int(config["seed"]) + 13,
                )
            )
            metrics["bootstrap_unit"] = "prompt"
        metrics["q_minus_anchor_ci"] = q_delta
        metrics["particle_oracle_minus_sampling_oracle_ci"] = oracle_delta
        metrics["particle_oracle_minus_matched_zero_latent_oracle_ci"] = (
            matched_oracle_delta
        )
        result[benchmark] = metrics
    return result


def stage_eval_math(config: Mapping[str, Any], *, checkpoint: Path | None) -> None:
    context = _distributed_context(config, require_distributed=True)
    try:
        _seed_everything(int(config["seed"]) + 601, context)
        memory_plan = _load_memory_plan(config, context)
        prompt_batch_size = int(memory_plan["rollout_prompt_batch_size"])
        _validated_data_fingerprints(config)
        checkpoint = _resolve_eval_checkpoint(config, checkpoint)
        policy, tokenizer = load_policy(config, context.device, load_warm_q=False)
        loaded = _load_inference_checkpoint(
            checkpoint, policy=policy, config=config, context=context
        )
        q_gate = _load_final_q_gate(config, checkpoint, task_type="math")
        records = _load_records(
            _project_path(config["paths"]["data_directory"]) / "eval_math.jsonl"
        )
        local_records = records[context.rank :: context.world_size]
        router = _reward_scorers(config, need_code=False)
        output_rows: list[dict[str, Any]] = []
        processed = 0
        for batch_number, record_batch in enumerate(
            _chunks(local_records, prompt_batch_size)
        ):
            common_seed = (
                int(config["seed"])
                + context.rank * 100_003
                + 607
                + batch_number * 1_009
            )
            particle_engine = _engine(
                policy,
                tokenizer,
                config,
                max_new_tokens=int(config["generation"]["math_eval_max_new_tokens"]),
                seed=common_seed,
                ordinary_sampling=False,
                compute_reference_logprobs=False,
            )
            ordinary_engine = _engine(
                policy,
                tokenizer,
                config,
                max_new_tokens=int(config["generation"]["math_eval_max_new_tokens"]),
                # Common token RNG with the particle arm makes branches 1..K-1
                # an exact paired zero-noise control. Fresh per-batch engines
                # prevent scale-dependent early EOS from desynchronizing later
                # prompt batches.
                seed=common_seed,
                ordinary_sampling=True,
                compute_reference_logprobs=False,
            )
            examples = [
                _rollout_example(record, config["prompting"])
                for record in record_batch
            ]
            particle = _rescore(
                particle_engine.generate(examples), examples, router=router
            )
            ordinary = _rescore(
                ordinary_engine.generate(examples), examples, router=router
            )
            with torch.no_grad():
                logits_batch = _q_logits(policy, particle).float().cpu()
                selected_batch = select_with_anchor_fallback(
                    logits_batch,
                    float(q_gate["margin"]),
                    bool(q_gate["ready"]),
                )
            particle_labels = _strict_q_labels(particle).cpu()
            ordinary_labels = _strict_q_labels(ordinary).cpu()
            for batch_index, record in enumerate(record_batch):
                selected = int(selected_batch[batch_index])
                majority_index, normalized = _math_equivalence_majority_vote(
                    ordinary.response_texts[batch_index]
                )
                particle_correct = particle_labels[batch_index]
                ordinary_correct = ordinary_labels[batch_index]
                matched_texts = [
                    particle.response_texts[batch_index][0],
                    *ordinary.response_texts[batch_index][1:],
                ]
                matched_correct = torch.cat(
                    (particle_correct[:1], ordinary_correct[1:])
                )
                matched_majority_index, matched_normalized = (
                    _math_equivalence_majority_vote(matched_texts)
                )
                output_rows.append(
                    {
                        "id": record["id"],
                        "benchmark": record["source"],
                        "cluster_id": str(
                            (record.get("metadata") or {}).get(
                                "original_id", record["id"]
                            )
                        ),
                        "checkpoint_step": int(loaded["step"]),
                        "anchor_correct": float(particle_correct[0]),
                        "ordinary_mean_correct": float(ordinary_correct.mean()),
                        "ordinary_oracle_correct": float(ordinary_correct.max()),
                        "ordinary_majority_correct": float(
                            ordinary_correct[majority_index]
                        ),
                        "matched_zero_latent_mean_correct": float(
                            matched_correct.mean()
                        ),
                        "matched_zero_latent_oracle_correct": float(
                            matched_correct.max()
                        ),
                        "matched_zero_latent_majority_correct": float(
                            matched_correct[matched_majority_index]
                        ),
                        "particle_mean_correct": float(particle_correct.mean()),
                        "particle_oracle_correct": float(particle_correct.max()),
                        "q_selected_correct": float(particle_correct[selected]),
                        "q_selected_branch": selected,
                        "q_logits": [
                            float(value) for value in logits_batch[batch_index]
                        ],
                        "ordinary_majority_branch": majority_index,
                        "matched_zero_latent_majority_branch": (
                            matched_majority_index
                        ),
                        "ordinary_normalized_answers": normalized,
                        "matched_zero_latent_normalized_answers": (
                            matched_normalized
                        ),
                        "ordinary_texts": ordinary.response_texts[batch_index],
                        "matched_zero_latent_texts": matched_texts,
                        "particle_texts": particle.response_texts[batch_index],
                        "ordinary_correctness": [
                            float(value) for value in ordinary_correct
                        ],
                        "particle_correctness": [
                            float(value) for value in particle_correct
                        ],
                    }
                )
            processed += len(record_batch)
            if context.is_main and processed % 25 < len(record_batch):
                print(
                    json.dumps({"math_eval_rank0_prompts": processed}), flush=True
                )
        eval_directory = _project_path(config["paths"]["run_directory"]) / "eval_math"
        shard = eval_directory / f"rank-{context.rank:02d}.jsonl"
        _write_jsonl_records(shard, output_rows)
        _barrier(context)
        if context.is_main:
            merged: list[dict[str, Any]] = []
            for path in sorted(eval_directory.glob("rank-*.jsonl")):
                with path.open("r", encoding="utf-8") as handle:
                    merged.extend(json.loads(line) for line in handle if line.strip())
            if len(merged) != len(records):
                raise RuntimeError(f"merged {len(merged)} math eval rows, expected {len(records)}")
            merged.sort(key=lambda row: (row["benchmark"], row["id"]))
            _write_jsonl_records(eval_directory / "all-results.jsonl", merged)
            summary = {
                "checkpoint": str(checkpoint),
                "checkpoint_step": int(loaded["step"]),
                "particle_mode": _particle_mode(config),
                "inference_stage": str(loaded["stage"]),
                "inference_provenance_hash": _canonical_hash(loaded["provenance"]),
                "inference_provenance": loaded["provenance"],
                "selected_relative_rms": loaded["provenance"].get(
                    "selected_relative_rms"
                ),
                "noise_scale_selection_sha256": loaded["provenance"].get(
                    "noise_scale_selection_sha256"
                ),
                "training_stopped_for_budget": bool(
                    loaded.get("stopped_for_budget", False)
                ),
                "q_gate": q_gate,
                "benchmarks": _aggregate_math_eval(merged, config),
                "ordinary_definition": "four stochastic zero-latent samples",
                "matched_zero_latent_definition": (
                    "one clean greedy anchor plus three stochastic zero-latent samples"
                ),
                "particle_definition": "one greedy z=0 anchor plus three particle explorers",
            }
            _atomic_json(eval_directory / "summary.json", summary)
            print(json.dumps({"math_evaluation": summary}, sort_keys=True), flush=True)
    finally:
        _destroy_process_group()


def _load_mbpp_plus(config: Mapping[str, Any]) -> tuple[list[tuple[str, str]], str]:
    expected_version = str(config["evaluation"]["evalplus_version"])
    observed_version = package_versions()["evalplus"]
    if observed_version != expected_version:
        raise RuntimeError(
            f"EvalPlus version mismatch: installed {observed_version}, expected {expected_version}"
        )
    try:
        from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    except ImportError as exc:
        raise RuntimeError(
            f"EvalPlus {expected_version} is required for MBPP+ prompt loading"
        ) from exc
    problems = get_mbpp_plus()
    rows: list[tuple[str, str]] = []
    for task_id in sorted(problems):
        problem = problems[task_id]
        prompt = problem.get("prompt") or problem.get("text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(f"MBPP+ task {task_id} has no prompt")
        rows.append((str(task_id), prompt.strip()))
    dataset_hash = str(get_mbpp_plus_hash())
    if not dataset_hash:
        raise RuntimeError("EvalPlus returned an empty MBPP+ dataset hash")
    return rows, dataset_hash


def _write_evalplus_samples(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task_id, solution in rows:
            handle.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")


def stage_eval_mbpp(config: Mapping[str, Any], *, checkpoint: Path | None) -> None:
    """Generate MBPP+ samples; execution is intentionally left to EvalPlus Docker."""

    context = _distributed_context(config, require_distributed=True)
    try:
        _seed_everything(int(config["seed"]) + 701, context)
        memory_plan = _load_memory_plan(config, context)
        prompt_batch_size = int(memory_plan["rollout_prompt_batch_size"])
        _validated_data_fingerprints(config)
        checkpoint = _resolve_eval_checkpoint(config, checkpoint)
        policy, tokenizer = load_policy(config, context.device, load_warm_q=False)
        loaded = _load_inference_checkpoint(
            checkpoint, policy=policy, config=config, context=context
        )
        q_gate = _load_final_q_gate(config, checkpoint, task_type="code")
        tasks, mbpp_dataset_hash = _load_mbpp_plus(config)
        local_tasks = tasks[context.rank :: context.world_size]
        rows: list[dict[str, Any]] = []
        suffix = str(config["prompting"]["code_suffix"])
        processed = 0
        for batch_number, task_batch in enumerate(
            _chunks(local_tasks, prompt_batch_size)
        ):
            common_seed = (
                int(config["seed"])
                + context.rank * 100_003
                + 709
                + batch_number * 1_009
            )
            particle_engine = _engine(
                policy,
                tokenizer,
                config,
                max_new_tokens=int(config["generation"]["code_eval_max_new_tokens"]),
                seed=common_seed,
                ordinary_sampling=False,
                compute_reference_logprobs=False,
            )
            ordinary_engine = _engine(
                policy,
                tokenizer,
                config,
                max_new_tokens=int(config["generation"]["code_eval_max_new_tokens"]),
                seed=common_seed,
                ordinary_sampling=True,
                compute_reference_logprobs=False,
            )
            examples = [
                RolloutExample(
                    prompt + suffix,
                    "",
                    task_id,
                    {"task_type": "code"},
                )
                for task_id, prompt in task_batch
            ]
            # Do not call a verifier here: EvalPlus hidden tests run later only
            # inside its official Docker image.
            particle = particle_engine.generate(examples)
            ordinary = ordinary_engine.generate(examples)
            with torch.no_grad():
                logits_batch = _q_logits(policy, particle).float().cpu()
                selected_batch = select_with_anchor_fallback(
                    logits_batch,
                    float(q_gate["margin"]),
                    bool(q_gate["ready"]),
                )
            for batch_index, (task_id, _) in enumerate(task_batch):
                particle_solutions = [
                    extract_python_code(text)
                    for text in particle.response_texts[batch_index]
                ]
                ordinary_solutions = [
                    extract_python_code(text)
                    for text in ordinary.response_texts[batch_index]
                ]
                rows.append(
                    {
                        "task_id": task_id,
                        "particle_solutions": particle_solutions,
                        "ordinary_solutions": ordinary_solutions,
                        "q_selected_branch": int(selected_batch[batch_index]),
                        "q_logits": [
                            float(value) for value in logits_batch[batch_index]
                        ],
                    }
                )
            processed += len(task_batch)
            if context.is_main and processed % 20 < len(task_batch):
                print(json.dumps({"mbpp_rank0_tasks": processed}), flush=True)
        output_directory = _project_path(config["paths"]["run_directory"]) / "evalplus_mbpp"
        _write_jsonl_records(output_directory / f"rank-{context.rank:02d}.jsonl", rows)
        _barrier(context)
        if context.is_main:
            merged: list[dict[str, Any]] = []
            for path in sorted(output_directory.glob("rank-*.jsonl")):
                with path.open("r", encoding="utf-8") as handle:
                    merged.extend(json.loads(line) for line in handle if line.strip())
            if len(merged) != len(tasks):
                raise RuntimeError(f"merged {len(merged)} MBPP+ tasks, expected {len(tasks)}")
            merged.sort(key=lambda row: row["task_id"])
            _write_evalplus_samples(
                output_directory / "ordinary-k4.jsonl",
                (
                    (row["task_id"], solution)
                    for row in merged
                    for solution in row["ordinary_solutions"]
                ),
            )
            _write_evalplus_samples(
                output_directory / "particle-k4.jsonl",
                (
                    (row["task_id"], solution)
                    for row in merged
                    for solution in row["particle_solutions"]
                ),
            )
            _write_evalplus_samples(
                output_directory / "particle-anchor.jsonl",
                ((row["task_id"], row["particle_solutions"][0]) for row in merged),
            )
            _write_evalplus_samples(
                output_directory / "matched-zero-latent-k4.jsonl",
                (
                    (row["task_id"], solution)
                    for row in merged
                    for solution in [
                        row["particle_solutions"][0],
                        *row["ordinary_solutions"][1:],
                    ]
                ),
            )
            _write_evalplus_samples(
                output_directory / "particle-q-selected.jsonl",
                (
                    (row["task_id"], row["particle_solutions"][row["q_selected_branch"]])
                    for row in merged
                ),
            )
            sample_names = (
                "ordinary-k4.jsonl",
                "particle-k4.jsonl",
                "particle-anchor.jsonl",
                "matched-zero-latent-k4.jsonl",
                "particle-q-selected.jsonl",
            )
            sample_sha256 = {
                name: sha256_file(output_directory / name) for name in sample_names
            }
            image = str(config["evaluation"]["evalplus_image"])
            commands = [
                "docker run --rm --pull=always --network none --pids-limit 512 "
                "--memory 32g --cpus 8 --cap-drop ALL "
                "--security-opt no-new-privileges -v \"$PWD:/app:rw\" "
                f"{image} "
                f"evalplus.evaluate --dataset mbpp --samples /app/{name}"
                for name in sample_names
            ]
            _atomic_json(
                output_directory / "generation-summary.json",
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": int(loaded["step"]),
                    "particle_mode": _particle_mode(config),
                    "inference_stage": str(loaded["stage"]),
                    "inference_provenance_hash": _canonical_hash(
                        loaded["provenance"]
                    ),
                    "inference_provenance": loaded["provenance"],
                    "selected_relative_rms": loaded["provenance"].get(
                        "selected_relative_rms"
                    ),
                    "noise_scale_selection_sha256": loaded["provenance"].get(
                        "noise_scale_selection_sha256"
                    ),
                    "training_stopped_for_budget": bool(
                        loaded.get("stopped_for_budget", False)
                    ),
                    "tasks": len(merged),
                    "task_ids": [str(row["task_id"]) for row in merged],
                    "mbpp_dataset_hash": mbpp_dataset_hash,
                    "evalplus_version": str(config["evaluation"]["evalplus_version"]),
                    "evalplus_image": image,
                    "sample_sha256": sample_sha256,
                    "expected_result_files": [
                        name.replace(".jsonl", "_eval_results.json")
                        for name in sample_names
                    ],
                    "q_gate": q_gate,
                    "docker_working_directory": str(output_directory),
                    "docker_commands": commands,
                    "safety": "Generated code has not been executed locally. Run only in EvalPlus Docker.",
                },
            )
            print(
                json.dumps(
                    {
                        "mbpp_generation": "complete",
                        "directory": str(output_directory),
                        "next": "run the five commands in generation-summary.json from that directory",
                    }
                ),
                flush=True,
            )
    finally:
        _destroy_process_group()


def validate_v1_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "model",
        "runtime",
        "memory",
        "paths",
        "prompting",
        "adapter",
        "particles",
        "generation",
        "q_warmup",
        "rl",
        "verification",
        "evaluation",
        "seed",
    }
    missing = required - set(config)
    if missing:
        raise ValueError("config is missing sections: " + ", ".join(sorted(missing)))
    runtime = config["runtime"]
    memory = config["memory"]
    model = config["model"]
    prompting = config["prompting"]
    particles = config["particles"]
    generation = config["generation"]
    q_warmup = config["q_warmup"]
    rl = config["rl"]
    verification = config["verification"]
    evaluation = config["evaluation"]

    world_size = int(runtime["world_size"])
    if world_size not in {2, 3}:
        raise ValueError("the tested V1 supports exactly two or three GPU ranks")
    if str(runtime["mixed_precision"]).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("runtime mixed_precision must be bfloat16")
    if str(runtime["dynamo_backend"]).lower() not in {"no", "none"}:
        raise ValueError("the hook-based V1 requires runtime dynamo_backend='no'")
    if not isinstance(runtime["deterministic_algorithms"], bool) or not isinstance(
        runtime["tf32"], bool
    ):
        raise ValueError("runtime deterministic_algorithms and tf32 must be booleans")
    for name in ("nccl_timeout_minutes", "checkpoint_every_updates", "log_every_updates"):
        if int(runtime[name]) <= 0:
            raise ValueError(f"runtime {name} must be positive")
    if float(runtime["stop_buffer_minutes"]) < 0.0:
        raise ValueError("runtime stop_buffer_minutes must be non-negative")

    if memory.get("auto_scale") is not True:
        raise ValueError("V1 requires memory auto_scale=true")
    target_fraction = float(memory["target_fraction"])
    hard_limit_fraction = float(memory["hard_limit_fraction"])
    if not 0.70 <= target_fraction <= 0.80:
        raise ValueError("memory target_fraction must lie in [0.70,0.80]")
    if not target_fraction <= hard_limit_fraction <= 0.80:
        raise ValueError(
            "memory hard_limit_fraction must be at least target_fraction and at most 0.80"
        )
    if float(memory["minimum_free_gib"]) <= 0.0:
        raise ValueError("memory minimum_free_gib must be positive")
    if float(memory["scaling_safety_factor"]) < 1.0:
        raise ValueError("memory scaling_safety_factor must be at least 1")
    target_per_rank = int(memory["target_prompt_groups_per_rank_update"])
    valid_batch_candidates(
        memory["candidate_prompt_batch_sizes"],
        target_prompts_per_rank_update=target_per_rank,
    )
    if int(memory["maximum_q_batch_prompt_groups"]) < 16:
        raise ValueError("memory maximum_q_batch_prompt_groups must be at least 16")
    if not str(config["paths"].get("memory_plan", "")):
        raise ValueError("paths memory_plan must be non-empty")

    if str(model["dtype"]).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("model dtype must be bfloat16")
    if str(model["revision"]) in {"", "main"}:
        raise ValueError("pin the HRM checkpoint revision; 'main' is not reproducible")
    if model.get("trust_remote_code") is not False:
        raise ValueError("native HRM V1 requires trust_remote_code=false")
    if str(model["attention_implementation"]).lower() != "sdpa":
        raise ValueError("the tested V1 attention implementation is sdpa")
    if int(model["expected_max_position_embeddings"]) <= 0:
        raise ValueError("expected_max_position_embeddings must be positive")

    for name in ("response_prefix", "math_suffix", "code_suffix"):
        if not str(prompting[name]):
            raise ValueError(f"prompting {name} must be non-empty")
    if "mode" not in particles:
        raise ValueError("particles mode must be explicitly configured")
    particle_mode = _particle_mode(config)
    if int(particles["count"]) != 4:
        raise ValueError("V1 is fixed at K=4")
    if int(particles["anchor_index"]) != 0:
        raise ValueError("branch zero must remain the exact anchor")
    if str(particles["anchor_decode"]).lower() != "greedy":
        raise ValueError("branch zero must use greedy anchor_decode")
    if float(particles["explorer_temperature"]) <= 0.0:
        raise ValueError("explorer_temperature must be positive")
    if float(particles["top_p"]) != 1.0:
        raise ValueError(
            "V1 requires top_p=1.0 for matched sampling and stable learned-mode PPO replay"
        )

    if particle_mode == "gaussian":
        if not str(config["paths"].get("noise_scale_selection", "")):
            raise ValueError("paths noise_scale_selection must be non-empty")
        gaussian = particles.get("gaussian")
        if not isinstance(gaussian, Mapping):
            raise ValueError("gaussian particle mode requires particles.gaussian")
        if gaussian.get("schedule") != "response_fixed":
            raise ValueError("gaussian schedule must be 'response_fixed'")
        raw_candidates = gaussian.get("scale_candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise ValueError("gaussian scale_candidates must contain zero and a positive scale")
        if any(isinstance(value, bool) for value in raw_candidates):
            raise ValueError("gaussian scale_candidates must be finite numbers")
        candidates = [float(value) for value in raw_candidates]
        if any(not math.isfinite(value) for value in candidates):
            raise ValueError("gaussian scale_candidates must be finite numbers")
        if candidates != sorted(set(candidates)) or candidates[0] != 0.0:
            raise ValueError(
                "gaussian scale_candidates must be sorted, unique, and begin with zero"
            )
        if candidates[-1] > float(config["adapter"]["max_relative_rms"]):
            raise ValueError("gaussian scale_candidates exceed adapter max_relative_rms")
        positive_candidates = [value for value in candidates if value > 0.0]
        if not positive_candidates:
            raise ValueError("gaussian scale_candidates need a positive scale")
        default_scale = float(gaussian["default_relative_rms"])
        if default_scale not in positive_candidates:
            raise ValueError(
                "gaussian default_relative_rms must be a declared positive candidate"
            )
        if int(gaussian["sweep_prompt_groups"]) <= 0:
            raise ValueError("gaussian sweep_prompt_groups must be positive")
        tie_tolerance = float(gaussian["oracle_tie_tolerance"])
        if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
            raise ValueError("gaussian oracle_tie_tolerance must be finite and non-negative")

    if generation.get("use_cache_for_rollout") is not True:
        raise ValueError("the budgeted V1 requires cached rollout generation")
    if generation.get("first_token_mode") != "causal_prefix":
        raise ValueError("the V1 requires first_token_mode='causal_prefix'")
    maximum_context = int(model["expected_max_position_embeddings"])
    for name in (
        "train_max_new_tokens",
        "q_collect_max_new_tokens",
        "math_eval_max_new_tokens",
        "code_eval_max_new_tokens",
    ):
        value = int(generation[name])
        if value <= 0 or value >= maximum_context:
            raise ValueError(f"generation {name} must lie in (0, model context length)")

    if rl.get("one_ppo_epoch_per_rollout") is not True:
        raise ValueError("V1 requires exactly one PPO epoch per fresh rollout")
    if rl.get("prompt_micro_batch_size") != "auto":
        raise ValueError("rl prompt_micro_batch_size must be auto")
    if rl.get("gradient_accumulation_steps") != "auto":
        raise ValueError("rl gradient_accumulation_steps must be auto")
    for name in ("max_updates", "pilot_updates"):
        if int(rl[name]) <= 0:
            raise ValueError(f"rl {name} must be positive")
    for name in ("actor_learning_rate", "q_learning_rate", "gradient_clip_norm"):
        if float(rl[name]) <= 0.0:
            raise ValueError(f"rl {name} must be positive")
    if not 0.0 <= float(rl["rescue_alpha"]) <= 1.0:
        raise ValueError("rescue_alpha must lie in [0,1]")
    if not 0.0 <= float(rl["code_fraction"]) < 1.0:
        raise ValueError("code_fraction must lie in [0,1)")
    if not 0.0 <= float(rl["code_binary_reward_weight"]) <= 1.0:
        raise ValueError("code_binary_reward_weight must lie in [0,1]")
    for name in ("code_min_nonzero_fraction", "code_min_mixed_fraction"):
        if not 0.0 <= float(rl[name]) <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")
    if int(rl["code_min_nonzero_groups"]) < 0:
        raise ValueError("code_min_nonzero_groups must be non-negative")

    groups = int(q_warmup["expected_prompt_groups"])
    candidates = int(q_warmup["target_candidate_examples"])
    if candidates != groups * int(particles["count"]):
        raise ValueError("Q candidate target must equal prompt groups times K")
    if not 5_000 <= candidates <= 10_000:
        raise ValueError("Q warmup must contain 5k-10k candidate examples")
    if int(q_warmup["epochs"]) <= 0:
        raise ValueError("Q epochs must be positive")
    if q_warmup.get("batch_prompt_groups") != "auto":
        raise ValueError("Q batch_prompt_groups must be auto")
    if not 0.0 < float(q_warmup["prior_probability"]) < 1.0:
        raise ValueError("Q prior_probability must lie in (0,1)")
    split_percentages = q_warmup.get("split_percentages")
    if not isinstance(split_percentages, Mapping) or set(split_percentages) != {
        "train",
        "early_stop",
        "margin_select",
        "safety_test",
    }:
        raise ValueError("Q split_percentages must define four disjoint splits")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in split_percentages.values()
    ) or sum(split_percentages.values()) != 100:
        raise ValueError("Q split_percentages must be positive and sum to 100")
    if int(q_warmup["bootstrap_samples"]) < 100:
        raise ValueError("Q bootstrap_samples must be at least 100")
    for name in (
        "minimum_global_calibration_prompts",
        "minimum_math_calibration_prompts",
        "minimum_code_calibration_prompts",
    ):
        if int(q_warmup[name]) <= 0:
            raise ValueError(f"Q {name} must be positive")

    adapter = config["adapter"]
    if not 0.0 < float(adapter["initial_relative_rms"]) < float(adapter["max_relative_rms"]):
        raise ValueError("initial particle RMS must be below its cap")
    if int(adapter["latent_size"]) <= 0 or int(adapter["bottleneck_size"]) <= 0:
        raise ValueError("adapter latent and bottleneck sizes must be positive")

    if verification.get("math_backend") != "math_verify":
        raise ValueError("verification math_backend must be math_verify")
    if verification.get("code_backend") != "open_r1_sandbox":
        raise ValueError("verification code_backend must be open_r1_sandbox")
    open_r1_commit = str(verification.get("open_r1_commit", ""))
    if len(open_r1_commit) != 40 or any(
        character not in "0123456789abcdef" for character in open_r1_commit
    ):
        raise ValueError("verification open_r1_commit must be a full lowercase Git SHA")
    if verification.get("code_provider") not in {"e2b", "morph"}:
        raise ValueError("verification code_provider must be e2b or morph")
    if verification.get("fail_closed_without_sandbox") is not True:
        raise ValueError("verification must fail closed without a sandbox")
    if int(verification["code_parallelism_per_rank"]) <= 0:
        raise ValueError("code_parallelism_per_rank must be positive")

    if evaluation.get("prompt_batch_size") != "auto":
        raise ValueError("evaluation prompt_batch_size must be auto")
    if int(evaluation["ordinary_sampling_candidates"]) != int(particles["count"]) or int(
        evaluation["particle_candidates"]
    ) != int(particles["count"]):
        raise ValueError("evaluation candidate counts must equal K")
    if evaluation.get("save_all_candidate_text") is not True:
        raise ValueError("the auditable V1 requires save_all_candidate_text=true")
    if list(evaluation["math_benchmarks"]) != ["gsm8k", "math500", "gsm_symbolic"]:
        raise ValueError("the V1 math benchmark list does not match the implemented evaluator")
    if list(evaluation["code_benchmarks"]) != ["mbpp"]:
        raise ValueError("the V1 code benchmark list does not match the implemented evaluator")
    if str(evaluation.get("evalplus_version")) != "0.3.1":
        raise ValueError("the tested V1 pins EvalPlus 0.3.1")
    evalplus_image = str(evaluation.get("evalplus_image", ""))
    prefix = "ganler/evalplus@sha256:"
    digest = evalplus_image.removeprefix(prefix)
    if not evalplus_image.startswith(prefix) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("evaluation evalplus_image must be pinned by a SHA256 digest")
    return {
        "world_size": world_size,
        "k": 4,
        "particle_mode": particle_mode,
        "q_candidates": candidates,
        "model_revision": model["revision"],
        "memory_batching": "empirical 75% target / 80% hard cap",
        "bf16_model_and_trainables": True,
        "trainable_particle_intervention": particle_mode == "learned",
        "training_path": (
            "fixed Gaussian scale selection plus Q-only SFT"
            if particle_mode == "gaussian"
            else "learned particle adapter plus Q and PPO"
        ),
        "fp32_sensitive_reductions_and_optimizer_state": True,
        "code_execution": (
            "remote sandbox for reward-labelled code; EvalPlus Docker for final evaluation"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "v1_2gpu_poc.yaml"),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate",
            "nccl_probe",
            "shape_smoke",
            "ddp_smoke",
            "vram_plan",
            "noise_scale_sweep",
            "collect_q",
            "q_warmup",
            "rl_pilot",
            "rl",
            "final_q_calibration",
            "eval_math",
            "eval_mbpp",
        ),
    )
    parser.add_argument("--resume-from")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--replan-memory",
        action="store_true",
        help="replace a compatible saved plan before Q/training artifacts exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve()
        config = _read_yaml(config_path)
        summary = validate_v1_config(config)
        if args.stage == "validate":
            print(json.dumps({"config": str(config_path), "validated": summary}, sort_keys=True))
            return 0
        if args.resume_from and args.stage != "rl":
            raise ValueError("--resume-from is valid only for the full rl stage")
        if args.checkpoint and args.stage not in {
            "final_q_calibration",
            "eval_math",
            "eval_mbpp",
        }:
            raise ValueError("--checkpoint is valid only for final calibration/evaluation")
        if args.replan_memory and args.stage != "vram_plan":
            raise ValueError("--replan-memory is valid only for the vram_plan stage")
        checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
        resume = Path(args.resume_from).expanduser().resolve() if args.resume_from else None
        stages = {
            "nccl_probe": lambda: stage_nccl_probe(config),
            "shape_smoke": lambda: stage_shape_smoke(config),
            "ddp_smoke": lambda: stage_ddp_smoke(config),
            "vram_plan": lambda: stage_vram_plan(config, force=args.replan_memory),
            "noise_scale_sweep": lambda: stage_noise_scale_sweep(config),
            "collect_q": lambda: stage_collect_q(config),
            "q_warmup": lambda: stage_q_warmup(config),
            "rl_pilot": lambda: stage_rl(config, pilot=True, resume_from=None),
            "rl": lambda: stage_rl(config, pilot=False, resume_from=resume),
            "final_q_calibration": lambda: stage_final_q_calibration(
                config, checkpoint=checkpoint
            ),
            "eval_math": lambda: stage_eval_math(config, checkpoint=checkpoint),
            "eval_mbpp": lambda: stage_eval_mbpp(config, checkpoint=checkpoint),
        }
        stages[args.stage]()
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


__all__ = [
    "RLTrainModule",
    "load_policy",
    "main",
    "stage_collect_q",
    "stage_ddp_smoke",
    "stage_eval_math",
    "stage_eval_mbpp",
    "stage_final_q_calibration",
    "stage_nccl_probe",
    "stage_noise_scale_sweep",
    "stage_q_warmup",
    "stage_rl",
    "stage_shape_smoke",
    "stage_vram_plan",
    "validate_v1_config",
]
