"""Adapter-only verifier-RL trainer with a separately supervised Q head."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

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
    score_actions,
)


@dataclass(frozen=True)
class TrainerConfig:
    alpha_rescue: float = 0.2
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.02
    injection_penalty_coefficient: float = 1e-3
    q_ranking_weight: float = 0.1
    max_grad_norm: float = 1.0
    target_kl_per_token: Optional[float] = None  # diagnostic target, not actor reward
    strict_adapter_only: bool = True
    allowed_actor_name_fragments: Tuple[str, ...] = ("adapter",)


@dataclass(frozen=True)
class TrainStepResult:
    rollout: ParticleRollout
    metrics: Dict[str, float]


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def validate_adapter_only_scope(
    model: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    *,
    q_optimizer: Optional[torch.optim.Optimizer] = None,
    allowed_name_fragments: Sequence[str] = ("adapter",),
) -> None:
    """Fail fast if PPO could mutate the backbone, embeddings, or LM head."""

    optimized = {id(parameter) for parameter in _optimizer_parameters(actor_optimizer)}
    q_optimized = (
        {id(parameter) for parameter in _optimizer_parameters(q_optimizer)}
        if q_optimizer is not None
        else set()
    )
    overlap = optimized & q_optimized
    if overlap:
        raise ValueError("actor and Q optimizers must have disjoint parameter sets")
    if not optimized:
        raise ValueError("actor optimizer has no parameters")
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    unknown = optimized - names_by_id.keys()
    if unknown:
        raise ValueError("actor optimizer contains parameters outside the policy model")
    disallowed = [
        names_by_id[parameter_id]
        for parameter_id in optimized
        if not any(fragment in names_by_id[parameter_id] for fragment in allowed_name_fragments)
    ]
    if disallowed:
        raise ValueError(
            "adapter-only POC refuses to optimize non-adapter parameters: "
            + ", ".join(sorted(disallowed))
        )

    # A forgotten requires_grad flag can accumulate gradients even if it is not
    # in this optimizer and makes experiment provenance ambiguous.
    unexpectedly_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in optimized | q_optimized
    ]
    if unexpectedly_trainable:
        raise ValueError(
            "freeze every policy parameter outside the particle adapter: "
            + ", ".join(sorted(unexpectedly_trainable))
        )


def _q_logits(q_head: nn.Module, terminal_states: Tensor, prompt_summary: Tensor) -> Tensor:
    """Invoke either ``q(terminal, prompt)`` or a keyword-based Q module."""

    try:
        q_parameter = next(q_head.parameters())
    except StopIteration:
        q_parameter = None
    if q_parameter is not None:
        terminal_states = terminal_states.to(
            device=q_parameter.device, dtype=q_parameter.dtype
        )
        prompt_summary = prompt_summary.to(
            device=q_parameter.device, dtype=q_parameter.dtype
        )
    try:
        logits = q_head(terminal_states, prompt_summary)
    except TypeError:
        logits = q_head(terminal_state=terminal_states, prompt_summary=prompt_summary)
    if isinstance(logits, dict):
        logits = logits.get("logits", logits.get("q_logits"))
    elif hasattr(logits, "logits"):
        logits = logits.logits
    if logits is None:
        raise RuntimeError("Q head did not return logits")
    if logits.ndim == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if logits.ndim != 1:
        raise RuntimeError(f"Q head must return one logit per candidate, got {logits.shape}")
    return logits


def _q_correctness_labels(rollout: ParticleRollout) -> Tensor:
    """Return and validate the strict correctness target used only by Q."""

    labels = rollout.q_labels if rollout.q_labels is not None else rollout.rewards
    if labels.ndim != 2 or labels.shape != rollout.rewards.shape:
        raise ValueError("rollout q_labels must have the same [batch, K] shape as rewards")
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("rollout q_labels must contain only finite values")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("rollout q_labels must contain only binary values 0 or 1")
    return labels


def _injection_penalty(
    output: Any, rollout: ParticleRollout, differentiable_zero: Tensor
) -> Tensor:
    if isinstance(output, dict):
        relative_rms = output.get("relative_rms")
        delta = output.get("injection_delta")
    else:
        relative_rms = getattr(output, "relative_rms", None)
        delta = getattr(output, "injection_delta", None)
    b, k, sequence = rollout.particle_mask.shape
    explorer_positions = rollout.particle_mask.reshape(b * k, sequence).clone()
    explorer_positions.reshape(b, k, sequence)[:, 0] = False
    if relative_rms is not None:
        values = relative_rms.float()
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.shape == explorer_positions.shape and bool(explorer_positions.any()):
            return values[explorer_positions].square().mean()
        raise RuntimeError("model relative_rms does not align with rollout.particle_mask")
    if delta is not None:
        values = delta.float().square().mean(dim=-1)
        if values.shape == explorer_positions.shape and bool(explorer_positions.any()):
            return values[explorer_positions].mean()
        raise RuntimeError("model injection_delta does not align with rollout.particle_mask")
    return differentiable_zero.sum() * 0.0


class ParticleTrainer:
    """One-rollout/one-epoch PPO updates for the small particle adapter.

    Q is updated from detached terminal states and external verifier labels in a
    separate optimizer step.  It is never called while constructing the actor
    reward or advantage.
    """

    def __init__(
        self,
        model: nn.Module,
        rollout_engine: ParticleRolloutEngine,
        actor_optimizer: torch.optim.Optimizer,
        q_optimizer: torch.optim.Optimizer,
        *,
        q_head: Optional[nn.Module] = None,
        config: Optional[TrainerConfig] = None,
    ) -> None:
        self.model = model
        self.q_head = q_head if q_head is not None else getattr(model, "q_head", None)
        if self.q_head is None:
            raise ValueError("provide q_head or construct the model with model.q_head")
        self.rollout_engine = rollout_engine
        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.config = config or TrainerConfig()
        if (
            self.config.target_kl_per_token is not None
            and self.config.target_kl_per_token <= 0
        ):
            raise ValueError("target_kl_per_token must be positive when provided")
        self.step = 0
        # The frozen policy backbone remains in deterministic inference mode
        # during both rollout and replay. Eval mode does not disable gradients
        # through the trainable adapter.
        self.model.eval()
        if self.config.strict_adapter_only:
            validate_adapter_only_scope(
                model,
                actor_optimizer,
                q_optimizer=q_optimizer,
                allowed_name_fragments=self.config.allowed_actor_name_fragments,
            )

    def train_batch(self, examples: Sequence[RolloutExample]) -> TrainStepResult:
        rollout = self.rollout_engine.generate(examples)
        metrics = self.train_rollout(rollout)
        return TrainStepResult(rollout=rollout, metrics=metrics)

    def train_rollout(self, rollout: ParticleRollout) -> Dict[str, float]:
        return self.train_rollouts([rollout])

    def train_accumulated_batches(
        self, batches: Sequence[Sequence[RolloutExample]]
    ) -> tuple[list[ParticleRollout], Dict[str, float]]:
        """Generate microbatches and perform one true accumulated optimizer step."""

        if not batches:
            raise ValueError("at least one microbatch is required")
        rollouts = [self.rollout_engine.generate(batch) for batch in batches]
        return rollouts, self.train_rollouts(rollouts)

    def train_rollouts(self, rollouts: Sequence[ParticleRollout]) -> Dict[str, float]:
        """Accumulate actor and Q gradients over independently generated batches."""

        if not rollouts:
            raise ValueError("at least one rollout is required")
        config = self.config
        self.model.eval()
        self.q_head.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.q_optimizer.zero_grad(set_to_none=True)
        total_prompts = float(sum(rollout.batch_size for rollout in rollouts))
        scalar_sums: Dict[str, float] = {}
        all_rewards: list[Tensor] = []
        all_q_labels: list[Tensor] = []
        all_q_logits: list[Tensor] = []

        for rollout in rollouts:
            accumulation_weight = rollout.batch_size / total_prompts
            rewards = rollout.rewards
            if rewards.ndim != 2 or rewards.shape[1] != rollout.k:
                raise ValueError("rollout rewards must have shape [batch, K]")
            all_rewards.append(rewards.detach())
            q_labels = _q_correctness_labels(rollout)
            all_q_labels.append(q_labels.detach())
            # This is the only actor credit source: external verifier rewards.
            advantages = anchor_rescue_advantages(rewards, alpha=config.alpha_rescue)
            actor_mask = rollout.action_mask.clone()
            actor_mask[:, 0] = False  # z0 is fixed and never an actor.
            replay = score_actions(self.model, rollout)
            policy = clipped_token_policy_loss(
                replay.logprobs,
                rollout.old_logprobs,
                advantages,
                actor_mask,
                clip_epsilon=config.clip_epsilon,
                reference_logprobs=rollout.reference_logprobs,
                kl_logprobs=replay.raw_logprobs,
                kl_coefficient=config.kl_coefficient,
            )
            injection_penalty = _injection_penalty(replay.output, rollout, replay.logprobs)
            actor_loss = policy.loss + config.injection_penalty_coefficient * injection_penalty
            (actor_loss * accumulation_weight).backward()

            # Q uses detached states and strict binary correctness labels.  The
            # actor may independently consume a shaped verifier reward.
            b, k, hidden = rollout.terminal_states.shape
            terminal = rollout.terminal_states.detach().reshape(b * k, hidden)
            prompt = rollout.prompt_summary.detach().reshape(b * k, -1)
            q_logits = _q_logits(self.q_head, terminal, prompt).reshape(b, k)
            all_q_logits.append(q_logits.detach().to(device=q_labels.device))
            q_loss = supervised_q_loss(
                q_logits,
                q_labels.detach().to(device=q_logits.device),
                ranking_weight=config.q_ranking_weight,
            )
            (q_loss.loss * accumulation_weight).backward()

            values = {
                "actor/loss": actor_loss,
                "actor/policy_loss": policy.policy_loss,
                "actor/reference_kl": policy.approx_kl,
                "actor/clip_fraction": policy.clip_fraction,
                "actor/mean_ratio": policy.mean_ratio,
                "actor/injection_penalty": injection_penalty,
                "q/loss": q_loss.loss,
                "q/bce": q_loss.bce_loss,
                "q/ranking": q_loss.ranking_loss,
            }
            for name, value in values.items():
                scalar_sums[name] = scalar_sums.get(name, 0.0) + (
                    float(value.detach()) * accumulation_weight
                )
            scalar_sums["q/ranking_pairs"] = scalar_sums.get("q/ranking_pairs", 0.0) + float(
                q_loss.ranking_pairs
            )

        actor_parameters = _optimizer_parameters(self.actor_optimizer)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_parameters, config.max_grad_norm, error_if_nonfinite=True
        )
        q_parameters = _optimizer_parameters(self.q_optimizer)
        q_grad_norm = torch.nn.utils.clip_grad_norm_(
            q_parameters, config.max_grad_norm, error_if_nonfinite=True
        )
        # Validate both independent gradient sets before mutating either module;
        # a non-finite Q update must not leave a half-applied actor checkpoint.
        self.actor_optimizer.step()
        self.q_optimizer.step()

        self.step += 1
        rewards = torch.cat(all_rewards, dim=0)
        q_labels = torch.cat(all_q_labels, dim=0)
        q_logits_before_update = torch.cat(all_q_logits, dim=0)
        explorer_rewards = rewards[:, 1:]
        mixed = ((rewards.sum(dim=1) > 0) & (rewards.sum(dim=1) < rewards.shape[1])).float()
        anchor_wrong = rewards[:, 0] < 0.5
        rescued = (explorer_rewards.sum(dim=1) > 0) & anchor_wrong
        rescue_rate = (
            rescued[anchor_wrong].float().mean() if bool(anchor_wrong.any()) else rewards.new_tensor(0.0)
        )
        oracle = rewards.max(dim=1).values.mean()
        q_selected_before_update = rewards.gather(
            1, q_logits_before_update.argmax(dim=1, keepdim=True)
        ).mean()

        q_explorer_labels = q_labels[:, 1:].float()
        q_labels_float = q_labels.float()
        q_mixed = (
            (q_labels_float.sum(dim=1) > 0)
            & (q_labels_float.sum(dim=1) < q_labels.shape[1])
        ).float()
        q_anchor_wrong = q_labels[:, 0] == 0
        q_rescued = (q_explorer_labels.sum(dim=1) > 0) & q_anchor_wrong
        q_rescue_rate = (
            q_rescued[q_anchor_wrong].float().mean()
            if bool(q_anchor_wrong.any())
            else q_labels_float.new_tensor(0.0)
        )
        q_oracle = q_labels_float.max(dim=1).values.mean()
        q_selected_correctness = q_labels_float.gather(
            1, q_logits_before_update.argmax(dim=1, keepdim=True)
        ).mean()
        reward_label_disagreement = (
            ~torch.isclose(rewards.float(), q_labels_float, rtol=0.0, atol=1e-7)
        ).float().mean()

        metrics = dict(scalar_sums)
        metrics.update({
            "step": float(self.step),
            "actor/grad_norm": float(actor_grad_norm.detach()),
            "q/grad_norm": float(q_grad_norm.detach()),
            "reward/anchor_accuracy": float(rewards[:, 0].mean()),
            "reward/explorer_accuracy": float(explorer_rewards.mean()),
            "reward/oracle_pass_at_k": float(oracle),
            "reward/q_selected_accuracy_preupdate": float(q_selected_before_update),
            "reward/rescue_given_anchor_wrong": float(rescue_rate),
            "groups/mixed_fraction": float(mixed.mean()),
            # Explicit aliases distinguish possibly shaped actor rewards from
            # the strict binary correctness targets used to supervise Q.
            "actor/reward_anchor_mean": float(rewards[:, 0].float().mean()),
            "actor/reward_explorer_mean": float(explorer_rewards.float().mean()),
            "actor/reward_oracle_at_k": float(rewards.float().max(dim=1).values.mean()),
            "q/correctness_anchor_accuracy": float(q_labels_float[:, 0].mean()),
            "q/correctness_explorer_accuracy": float(q_explorer_labels.mean()),
            "q/correctness_oracle_pass_at_k": float(q_oracle),
            "q/correctness_selected_accuracy_preupdate": float(q_selected_correctness),
            "q/correctness_rescue_given_anchor_wrong": float(q_rescue_rate),
            "q/correctness_mixed_fraction": float(q_mixed.mean()),
            "q/correctness_actor_reward_disagreement_fraction": float(
                reward_label_disagreement
            ),
        })
        if config.target_kl_per_token is not None:
            target = float(config.target_kl_per_token)
            metrics["actor/target_kl_per_token"] = target
            metrics["actor/kl_over_target"] = metrics["actor/reference_kl"] / target
        return metrics


def _project_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[2] / candidate
    return candidate.resolve()


def _dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported model dtype {name!r}") from exc


def build_components_from_config(config: Mapping[str, Any]):
    """Load the real HRM checkpoint and construct only the small trainable heads.

    Imports are local so the offline unit suite never imports Transformers or
    touches the network.
    """

    runtime = config.get("runtime", {})
    if runtime.get("dummy"):
        raise ValueError(
            "runtime.dummy is exercised by scripts/run_smoke.py; real train_from_config "
            "requires a Hugging Face checkpoint"
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .adapter import ParticleAdapterConfig
    from .model import ParticleHrmForCausalLM

    model_config = dict(config.get("model", {}))
    adapter_config = dict(config.get("adapter", {}))
    q_config = dict(config.get("q_head", {}))
    model_name = model_config.get("pretrained_model_name_or_path")
    if not model_name:
        raise ValueError("model.pretrained_model_name_or_path is required")
    if model_config.get("freeze_backbone") is not True:
        raise ValueError("POC-A is adapter-only; model.freeze_backbone must be true")
    revision = model_config.get("revision", "main")
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    dtype = _dtype_from_name(str(model_config.get("dtype", "bfloat16")))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id or pad_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = dict(
        revision=revision,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except TypeError:
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()
    hidden_size = int(getattr(base_model.config, "hidden_size"))
    particle_config = ParticleAdapterConfig(
        hidden_size=hidden_size,
        latent_size=int(adapter_config.get("latent_size", 64)),
        bottleneck_size=int(adapter_config.get("bottleneck_size", 64)),
        max_relative_rms=float(adapter_config.get("max_relative_rms", 0.10)),
        initial_relative_rms=float(adapter_config.get("initial_relative_rms", 0.03)),
        rms_eps=float(adapter_config.get("rms_epsilon", 1e-6)),
        detach_query=bool(adapter_config.get("detach_query_summary", True)),
    )
    model = ParticleHrmForCausalLM(
        base_model,
        adapter_config=particle_config,
        q_bottleneck_size=int(q_config.get("bottleneck_size", 256)),
        freeze_backbone=True,
    )
    device_name = str(model_config.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    model.to(device=device)
    # Keep small trainable modules in fp32 for optimizer/Q numerical stability;
    # ParticleAdapter and trainer cast their inputs explicitly.
    model.adapter.float()
    model.q_head.float()
    model.eval()
    return model, tokenizer


def _rollout_engine_from_config(
    model: nn.Module,
    tokenizer: Any,
    config: Mapping[str, Any],
    *,
    require_k4: bool = True,
) -> ParticleRolloutEngine:
    particles = dict(config.get("particles", {}))
    generation = dict(config.get("generation", {}))
    prompting = dict(config.get("prompting", {}))
    adapter = dict(config.get("adapter", {}))
    k = int(particles.get("count", 4))
    if require_k4 and k != 4:
        raise ValueError("POC v1 training is intentionally fixed at K=4")
    if k < 2:
        raise ValueError("evaluation requires at least one anchor and one explorer")
    if int(particles.get("anchor_index", 0)) != 0:
        raise ValueError("particle zero must be the anchor")
    return ParticleRolloutEngine(
        model,
        tokenizer,
        config=RolloutConfig(
            k=k,
            latent_dim=int(adapter.get("latent_size", 64)),
            max_new_tokens=int(generation.get("max_new_tokens", 128)),
            temperature=float(generation.get("explorer_temperature", 0.8)),
            top_p=float(generation.get("top_p", 1.0)),
            response_prefix=str(generation.get("response_prefix", "\nSolution:\n")),
            first_token_mode=str(generation.get("first_token_mode", "causal_prefix")),
            use_cache=bool(generation.get("use_cache_for_rollout", True)),
            compute_reference_logprobs=True,
            seed=int(config.get("seed", 0)),
            condition=str(prompting.get("condition", "synth,cot")),
        ),
    )


def _resume_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    # Exclude only operational sections that do not change the learned process.
    # All architecture, data-order, rollout, credit, optimizer, and Q settings
    # remain part of exact-resume compatibility.
    included = (
        "model",
        "prompting",
        "adapter",
        "particles",
        "generation",
        "data",
        "optimization",
        "q_head",
        "seed",
    )
    return {name: config.get(name) for name in included}


def _file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def train_from_config(
    config: Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    resume_from: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the budgeted adapter-only POC from a validated YAML mapping."""

    from .checkpoint import load_checkpoint, save_checkpoint
    from .data import load_jsonl

    optimization = dict(config.get("optimization", {}))
    q_config = dict(config.get("q_head", {}))
    data_config = dict(config.get("data", {}))
    if int(optimization.get("actor_epochs_per_rollout", 1)) != 1:
        raise ValueError("POC requires exactly one actor epoch per rollout")
    if q_config.get("use_q_as_actor_reward", False):
        raise ValueError("Q must never be used as actor reward")
    if float(config.get("generation", {}).get("top_p", 1.0)) != 1.0:
        raise ValueError(
            "actor training requires generation.top_p=1.0; dynamic nucleus support is "
            "unsafe for PPO/reference replay (evaluation may still sweep top_p)"
        )
    if int(config.get("particles", {}).get("count", 4)) != 4:
        raise ValueError("POC v1 training is intentionally fixed at K=4")
    if dry_run:
        return {"status": "validated", "network_or_model_loaded": False}

    destination = _project_path(
        output_dir or config.get("output", {}).get("directory", "runs/poc")
    )
    destination.mkdir(parents=True, exist_ok=True)
    occupied = (
        (destination / "metrics.jsonl").exists()
        or (destination / "resolved-config.json").exists()
        or any(destination.glob("checkpoint-*.pt"))
    )
    if occupied and resume_from is None:
        raise ValueError(
            f"refusing to append a new run into occupied output directory: {destination}"
        )
    if occupied and resume_from is not None and (destination / "resolved-config.json").exists():
        existing_config = json.loads(
            (destination / "resolved-config.json").read_text(encoding="utf-8")
        )
        if _resume_signature(existing_config) != _resume_signature(config):
            raise ValueError("occupied resume output belongs to an incompatible run")

    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    model, tokenizer = build_components_from_config(config)
    engine = _rollout_engine_from_config(model, tokenizer, config)
    weight_decay = float(optimization.get("weight_decay", 0.0))
    actor_optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=float(optimization.get("adapter_learning_rate", 1e-4)),
        weight_decay=weight_decay,
    )
    q_optimizer = torch.optim.AdamW(
        model.q_head.parameters(),
        lr=float(q_config.get("learning_rate", 3e-4)),
        weight_decay=float(q_config.get("weight_decay", weight_decay)),
    )
    trainer = ParticleTrainer(
        model,
        engine,
        actor_optimizer,
        q_optimizer,
        q_head=model.q_head,
        config=TrainerConfig(
            alpha_rescue=float(optimization.get("rescue_alpha", 0.2)),
            clip_epsilon=float(optimization.get("ppo_clip_epsilon", 0.2)),
            kl_coefficient=float(optimization.get("kl_coefficient", 0.02)),
            injection_penalty_coefficient=float(
                optimization.get("injection_penalty_coefficient", 1e-3)
            ),
            q_ranking_weight=float(q_config.get("ranking_weight", 0.1)),
            max_grad_norm=float(optimization.get("gradient_clip_norm", 1.0)),
            target_kl_per_token=(
                float(optimization["target_kl_per_token"])
                if "target_kl_per_token" in optimization
                else None
            ),
        ),
    )

    data_directory = _project_path(str(data_config.get("directory", "data/processed")))
    train_path = data_directory / str(data_config.get("train_file", "train.jsonl"))
    records = load_jsonl(train_path)
    maximum = int(data_config.get("max_train_examples", len(records)))
    records = records[:maximum]
    if not records:
        raise ValueError(f"training dataset is empty: {train_path}")
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "not-installed"
    provenance = {
        "created_unix": time.time(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "model": config.get("model", {}).get("pretrained_model_name_or_path"),
        "revision": config.get("model", {}).get("revision", "main"),
        "resolved_revision": getattr(
            getattr(getattr(model, "base_model", None), "config", None),
            "_commit_hash",
            None,
        ),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_file": str(train_path),
        "train_file_sha256": _file_sha256(train_path),
        "train_record_count": len(records),
        "manifest_sha256": _file_sha256(data_directory / "manifest.json"),
        "resume_signature": _resume_signature(config),
        "effective_training": {
            "adapter_learning_rate": float(
                optimization.get("adapter_learning_rate", 1e-4)
            ),
            "q_learning_rate": float(q_config.get("learning_rate", 3e-4)),
            "kl_coefficient": float(optimization.get("kl_coefficient", 0.02)),
            "target_kl_per_token": optimization.get("target_kl_per_token"),
            "injection_penalty_coefficient": float(
                optimization.get("injection_penalty_coefficient", 1e-3)
            ),
            "gradient_accumulation_steps": int(
                optimization.get("gradient_accumulation_steps", 1)
            ),
        },
    }
    if resume_from is None:
        (destination / "resolved-config.json").write_text(
            json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (destination / "run-metadata.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    resume_metadata: dict[str, Any] = {}
    if resume_from is not None:
        loaded = load_checkpoint(
            resume_from,
            adapter=model.adapter,
            q_head=model.q_head,
            actor_optimizer=actor_optimizer,
            q_optimizer=q_optimizer,
            rollout_engine=engine,
            map_location=next(model.adapter.parameters()).device,
        )
        trainer.step = int(loaded["step"])
        loaded_config = loaded.get("config")
        if loaded_config is None or _resume_signature(loaded_config) != _resume_signature(config):
            raise ValueError("resume checkpoint is incompatible with current model/adapter config")
        resume_metadata = dict(loaded.get("metadata", {}))
        if "loop_state" not in resume_metadata:
            raise ValueError("resume checkpoint lacks exact dataset-loop cursor state")
        prior_provenance = dict(resume_metadata.get("provenance", {}))
        for field in ("train_file_sha256", "train_record_count", "manifest_sha256"):
            if prior_provenance.get(field) != provenance.get(field):
                raise ValueError(f"resume dataset provenance changed: {field}")
        prior_commit = prior_provenance.get("resolved_revision")
        current_commit = provenance.get("resolved_revision")
        if prior_commit is not None and current_commit != prior_commit:
            raise ValueError("resolved Hugging Face model revision changed since checkpoint")
        if not (destination / "resolved-config.json").exists():
            (destination / "resolved-config.json").write_text(
                json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if not (destination / "run-metadata.json").exists():
            (destination / "run-metadata.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        with (destination / "resume-events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "resumed_unix": time.time(),
                        "checkpoint": str(Path(resume_from).resolve()),
                        "step": trainer.step,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    rounds = int(optimization.get("rollout_rounds", 1))
    micro_size = int(optimization.get("prompt_micro_batch_size", 1))
    accumulation = int(optimization.get("gradient_accumulation_steps", 1))
    if micro_size <= 0 or accumulation <= 0 or rounds <= 0:
        raise ValueError("rollout rounds, microbatch, and accumulation must be positive")
    checkpointing = dict(config.get("checkpointing", {}))
    if checkpointing.get("save_optimizer", True) is not True:
        raise ValueError("exact resume requires checkpointing.save_optimizer=true")
    if checkpointing.get("save_rollout_rng_state", True) is not True:
        raise ValueError("exact resume requires checkpointing.save_rollout_rng_state=true")
    save_every = int(checkpointing.get("save_every_updates", 50))
    log_every = int(config.get("logging", {}).get("log_every_updates", 1))
    if log_every <= 0:
        raise ValueError("logging.log_every_updates must be positive")
    deadline = float(os.environ.get("HRM_PARTICLE_DEADLINE_UNIX", "inf"))
    metrics_path = destination / "metrics.jsonl"
    rng = random.Random(seed)
    loop_state = dict(resume_metadata.get("loop_state", {}))
    if loop_state.get("python_rng_state") is not None:
        rng.setstate(loop_state["python_rng_state"])
    start_round = int(loop_state.get("round_index", 0))
    resume_next_start = int(loop_state.get("next_start", 0))
    resume_order_ids = loop_state.get("round_order_ids")
    stopped_for_budget = False
    last_metrics: Dict[str, float] = {}
    cursor_round = start_round
    cursor_next_start = resume_next_start
    cursor_order_ids = resume_order_ids

    def checkpoint_metadata() -> dict[str, Any]:
        return {
            "provenance": provenance,
            "stopped_for_budget": stopped_for_budget,
            "loop_state": {
                "round_index": cursor_round,
                "next_start": cursor_next_start,
                "round_order_ids": cursor_order_ids,
                "python_rng_state": rng.getstate(),
            },
        }

    records_by_id = {record.id: record for record in records}
    if trainer.step == 0 and resume_from is None:
        # Cheap, reproducible step-0 baseline for measuring what learning added
        # beyond the randomly initialized bounded adapter.
        save_checkpoint(
            destination / "checkpoint-initial.pt",
            adapter=model.adapter,
            q_head=model.q_head,
            step=0,
            actor_optimizer=actor_optimizer,
            q_optimizer=q_optimizer,
            rollout_engine=engine,
            config=config,
            metadata=checkpoint_metadata(),
        )
    for round_index in range(start_round, rounds):
        if round_index == start_round and resume_order_ids is not None:
            try:
                order = [records_by_id[record_id] for record_id in resume_order_ids]
            except KeyError as exc:
                raise ValueError("resume dataset no longer contains the saved record order") from exc
            range_start = resume_next_start
        else:
            order = list(records)
            if data_config.get("shuffle_train", True):
                rng.shuffle(order)
            range_start = 0
        cursor_round = round_index
        cursor_next_start = range_start
        cursor_order_ids = [record.id for record in order]
        effective = micro_size * accumulation
        for start in range(range_start, len(order), effective):
            if time.time() >= deadline:
                stopped_for_budget = True
                break
            chunk = order[start : start + effective]
            batches = [
                [
                    RolloutExample(
                        prompt=record.prompt,
                        answer=record.answer,
                        example_id=record.id,
                        metadata=record.metadata,
                    )
                    for record in chunk[offset : offset + micro_size]
                ]
                for offset in range(0, len(chunk), micro_size)
            ]
            _, last_metrics = trainer.train_accumulated_batches(batches)
            cursor_next_start = min(start + len(chunk), len(order))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            if trainer.step == 1 or trainer.step % log_every == 0:
                print(json.dumps({"training_metrics": last_metrics}, sort_keys=True), flush=True)
            if save_every > 0 and trainer.step % save_every == 0:
                save_checkpoint(
                    destination / f"checkpoint-{trainer.step:06d}.pt",
                    adapter=model.adapter,
                    q_head=model.q_head,
                    step=trainer.step,
                    actor_optimizer=actor_optimizer,
                    q_optimizer=q_optimizer,
                    rollout_engine=engine,
                    config=config,
                    metadata=checkpoint_metadata(),
                )
        if stopped_for_budget:
            break
        cursor_round = round_index + 1
        cursor_next_start = 0
        cursor_order_ids = None
        resume_order_ids = None
        resume_next_start = 0
    final_path = save_checkpoint(
        destination / "checkpoint-last.pt",
        adapter=model.adapter,
        q_head=model.q_head,
        step=trainer.step,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        rollout_engine=engine,
        config=config,
        metadata=checkpoint_metadata(),
    )
    return {
        "status": "budget_stop" if stopped_for_budget else "complete",
        "updates": trainer.step,
        "checkpoint": str(final_path),
        "last_metrics": last_metrics,
    }


__all__ = [
    "ParticleTrainer",
    "TrainerConfig",
    "TrainStepResult",
    "build_components_from_config",
    "train_from_config",
    "validate_adapter_only_scope",
]
