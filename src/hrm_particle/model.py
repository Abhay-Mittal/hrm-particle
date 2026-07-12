"""A scoped, validated H1-injection wrapper for Hugging Face HRM-Text.

The upstream HRM implementation reuses one ``H_module`` object once per high
cycle.  A temporary forward hook therefore provides a small, non-vendored
integration point: alter the output of the selected high cycle, then let all
subsequent L/H computation consume the altered state.  The hook is installed
for one forward only and removed in ``finally``.

This is deliberately a POC integration, not a general model patch.  Capability
validation fails loudly when the expected HRM structure is absent or when
execution modes that invalidate Python hook semantics are enabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import threading
from typing import Any

import torch
from torch import Tensor, nn

from .adapter import (
    ParticleAdapter,
    ParticleAdapterConfig,
    ParticleAdapterOutput,
    SharedQHead,
)


class HrmIntegrationError(RuntimeError):
    """Raised when a model/runtime cannot safely support the POC hook."""


@dataclass(frozen=True)
class HrmTextCapabilities:
    """Validated structural facts used by the wrapper."""

    core_path: str
    hidden_size: int
    high_cycles: int
    injection_after_high_cycle: int
    core_model: nn.Module
    high_module: nn.Module


@dataclass
class ParticleModelOutput:
    """Stable output contract used by rollout, PPO replay, and Q training."""

    logits: Tensor
    past_key_values: Any
    terminal_state: Tensor
    prompt_summary: Tensor | None
    q_logits: Tensor | None
    injection_delta: Tensor | None
    relative_rms: Tensor | None
    injection_amplitude: Tensor | None
    loss: Tensor | None = None
    hidden_states: Any = None
    attentions: Any = None
    base_output: Any = None

    @property
    def query_state(self) -> Tensor | None:
        """Alias used by cached rollout code."""

        return self.prompt_summary

    def to_tuple(self) -> tuple[Any, ...]:
        return (
            self.logits,
            self.past_key_values,
            self.terminal_state,
            self.prompt_summary,
            self.q_logits,
        )


def _model_children_for_search(module: nn.Module) -> list[tuple[str, nn.Module]]:
    children: list[tuple[str, nn.Module]] = []
    for attribute in ("model", "base_model", "module"):
        child = getattr(module, attribute, None)
        if isinstance(child, nn.Module) and child is not module:
            children.append((attribute, child))
    get_base_model = getattr(module, "get_base_model", None)
    if callable(get_base_model):
        try:
            child = get_base_model()
        except Exception:
            child = None
        if isinstance(child, nn.Module) and child is not module:
            children.append(("get_base_model()", child))
    return children


def _find_hrm_core(base_model: nn.Module) -> tuple[str, nn.Module]:
    queue: list[tuple[str, nn.Module, int]] = [("base_model", base_model, 0)]
    seen: set[int] = set()
    while queue:
        path, candidate, depth = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        required_attributes = ("H_module", "L_module", "embed_tokens")
        if all(hasattr(candidate, attribute) for attribute in required_attributes):
            return path, candidate
        if depth < 4:
            for child_name, child in _model_children_for_search(candidate):
                queue.append((f"{path}.{child_name}", child, depth + 1))
    raise HrmIntegrationError(
        "could not find an HRM-Text core exposing H_module, L_module, and embed_tokens; "
        "expected HrmTextForCausalLM.model (possibly beneath a PEFT/DDP wrapper)"
    )


def validate_hrm_text_model(
    base_model: nn.Module,
    *,
    injection_after_high_cycle: int = 0,
) -> HrmTextCapabilities:
    """Validate the exact upstream structure on which the hook relies."""

    if not isinstance(base_model, nn.Module):
        raise TypeError("base_model must be a torch.nn.Module")
    if hasattr(base_model, "_orig_mod"):
        raise HrmIntegrationError(
            "torch.compile/OptimizedModule is unsupported by the hook-based POC; "
            "wrap the eager model instead"
        )
    path, core = _find_hrm_core(base_model)
    config = getattr(core, "config", getattr(base_model, "config", None))
    if config is None:
        raise HrmIntegrationError("HRM core has no config")
    high_cycles = getattr(config, "H_cycles", None)
    hidden_size = getattr(config, "hidden_size", None)
    if not isinstance(high_cycles, int) or high_cycles < 2:
        raise HrmIntegrationError(
            f"expected integer config.H_cycles >= 2, got {high_cycles!r}"
        )
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise HrmIntegrationError(
            f"expected positive integer config.hidden_size, got {hidden_size!r}"
        )
    if not 0 <= injection_after_high_cycle < high_cycles - 1:
        raise HrmIntegrationError(
            "injection_after_high_cycle must leave at least one downstream high cycle; "
            f"got {injection_after_high_cycle} for H_cycles={high_cycles}"
        )
    high_module = getattr(core, "H_module")
    if not isinstance(high_module, nn.Module):
        raise HrmIntegrationError("core.H_module is not a torch.nn.Module")
    return HrmTextCapabilities(
        core_path=path,
        hidden_size=hidden_size,
        high_cycles=high_cycles,
        injection_after_high_cycle=injection_after_high_cycle,
        core_model=core,
        high_module=high_module,
    )


def _get_output_field(output: Any, name: str, default: Any = None) -> Any:
    if hasattr(output, name):
        return getattr(output, name)
    if isinstance(output, Mapping):
        return output.get(name, default)
    return default


def _align_token_mask(mask: Tensor, hidden_states: Tensor, *, name: str) -> Tensor:
    """Align a full-history mask to ``hidden_states[:, -seq_len:]``."""

    mask = torch.as_tensor(mask, device=hidden_states.device)
    if hidden_states.ndim != 3:
        raise HrmIntegrationError(
            f"captured H state must be [batch, sequence, hidden], got {tuple(hidden_states.shape)}"
        )
    if mask.ndim != 2 or mask.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"{name} must have shape [batch, sequence], got {tuple(mask.shape)}"
        )
    sequence_length = hidden_states.shape[1]
    if mask.shape[1] < sequence_length:
        raise ValueError(
            f"{name} length {mask.shape[1]} is shorter than hidden length {sequence_length}"
        )
    return mask[:, -sequence_length:].bool()


def _select_last_state(
    hidden_states: Tensor,
    mask: Tensor | None,
    *,
    fallback_mask: Tensor | None = None,
    name: str = "mask",
) -> Tensor:
    """Select the last true position per row, falling back safely per row."""

    if hidden_states.ndim == 2:
        return hidden_states
    if hidden_states.ndim != 3:
        raise HrmIntegrationError(
            f"captured H state must be rank 2 or 3, got {tuple(hidden_states.shape)}"
        )
    batch_size, sequence_length, _ = hidden_states.shape
    if mask is None:
        selected_mask = torch.ones(
            (batch_size, sequence_length), dtype=torch.bool, device=hidden_states.device
        )
    else:
        selected_mask = _align_token_mask(mask, hidden_states, name=name)
    rows_without_selection = ~selected_mask.any(dim=-1)
    if rows_without_selection.any() and fallback_mask is not None:
        fallback = _align_token_mask(fallback_mask, hidden_states, name="fallback_mask")
        selected_mask = torch.where(rows_without_selection[:, None], fallback, selected_mask)
        rows_without_selection = ~selected_mask.any(dim=-1)
    if rows_without_selection.any():
        selected_mask = selected_mask.clone()
        selected_mask[rows_without_selection, -1] = True

    positions = torch.arange(sequence_length, device=hidden_states.device).expand(batch_size, -1)
    last_positions = positions.masked_fill(~selected_mask, -1).max(dim=-1).values
    batch_positions = torch.arange(batch_size, device=hidden_states.device)
    return hidden_states[batch_positions, last_positions]


def _runtime_flag(module: nn.Module, name: str) -> bool:
    value = getattr(module, name, False)
    if callable(value):
        try:
            value = value()
        except TypeError:
            return False
    return bool(value)


class ParticleHrmForCausalLM(nn.Module):
    """Wrap an eager ``HrmTextForCausalLM`` with an H-cycle particle adapter.

    ``particle_mask`` (alias ``response_mask``) is mandatory whenever
    ``particle_z`` is supplied.  Cached response steps must also pass the fixed
    ``prompt_summary`` returned by a prior *clean* prefill as ``query_state``.

    PrefixLM first-token caveat
    ---------------------------
    A shared clean prompt prefill necessarily produces an unperturbed first
    sampled-token logit.  There are two correct ways to branch earlier:

    * After caching the *complete* official bidirectional prompt, append a truly
      causal, fixed response-prefix token and inject on that token.  This valid
      and efficient path branches the first sampled content-token logit.
    * With no fixed causal response prefix, run a clean summary pass and then
      recompute the full prompt per branch, injecting only on its final position.

    A token that belongs inside the official bidirectional prompt (for example
    its required end delimiter) must not be held out and later appended as if it
    were causal; that would change PrefixLM semantics.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter: nn.Module | None = None,
        q_head: SharedQHead | None = None,
        *,
        adapter_config: ParticleAdapterConfig | None = None,
        q_bottleneck_size: int | None = None,
        injection_after_high_cycle: int = 0,
        detach_q_state: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        capabilities = validate_hrm_text_model(
            base_model, injection_after_high_cycle=injection_after_high_cycle
        )
        self.base_model = base_model
        self._capabilities = capabilities
        if adapter is not None and adapter_config is not None:
            raise ValueError("pass either adapter or adapter_config, not both")
        created_adapter = adapter is None
        if adapter is None:
            adapter_config = adapter_config or ParticleAdapterConfig(
                hidden_size=capabilities.hidden_size
            )
            adapter = ParticleAdapter(adapter_config)
        if adapter.hidden_size != capabilities.hidden_size:
            raise ValueError(
                f"adapter hidden size {adapter.hidden_size} does not match HRM hidden size "
                f"{capabilities.hidden_size}"
            )
        self.adapter = adapter
        created_q_head = q_head is None
        self.q_head = q_head or SharedQHead(capabilities.hidden_size, q_bottleneck_size)
        if self.q_head.hidden_size != capabilities.hidden_size:
            raise ValueError("Q-head hidden size does not match HRM hidden size")
        # ``from_pretrained(..., device_map=...)`` may place the HRM before this
        # wrapper is constructed.  Put newly-created small modules next to the
        # H stack without changing their float32 parameter dtype.
        high_parameter = next(capabilities.high_module.parameters(), None)
        if high_parameter is not None:
            if created_adapter:
                self.adapter.to(device=high_parameter.device)
            if created_q_head:
                self.q_head.to(device=high_parameter.device)
        self.detach_q_state = bool(detach_q_state)
        self._forward_lock = threading.Lock()
        self._backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    @property
    def config(self) -> Any:
        return getattr(self.base_model, "config", self._capabilities.core_model.config)

    @property
    def capabilities(self) -> HrmTextCapabilities:
        return self._capabilities

    def freeze_backbone(self) -> None:
        """Freeze and keep the reference actor in evaluation mode."""

        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        self._backbone_frozen = True

    def train(self, mode: bool = True) -> "ParticleHrmForCausalLM":
        super().train(mode)
        if self._backbone_frozen:
            self.base_model.eval()
        return self

    def _validate_runtime(self) -> None:
        core = self._capabilities.core_model
        if _runtime_flag(self.base_model, "is_gradient_checkpointing") or _runtime_flag(
            core, "gradient_checkpointing"
        ):
            raise HrmIntegrationError(
                "gradient checkpointing is unsupported by the hook-based POC; the frozen "
                "backbone does not need it, so disable it before particle training"
            )
        dynamo = getattr(torch, "_dynamo", None)
        if dynamo is not None and dynamo.is_compiling():
            raise HrmIntegrationError(
                "torch.compile is unsupported by the hook-based POC; run the actor eagerly"
            )
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            raise HrmIntegrationError("TorchScript tracing/scripting is unsupported")

    @staticmethod
    def _resolve_aliases(
        particle_mask: Tensor | None,
        response_mask: Tensor | None,
        query_state: Tensor | None,
        prompt_summary: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if particle_mask is not None and response_mask is not None:
            if particle_mask.shape != response_mask.shape or not torch.equal(
                particle_mask.bool(), response_mask.bool()
            ):
                raise ValueError("particle_mask and response_mask aliases disagree")
        particle_mask = particle_mask if particle_mask is not None else response_mask
        if query_state is not None and prompt_summary is not None:
            if query_state.shape != prompt_summary.shape or not torch.equal(
                query_state, prompt_summary
            ):
                raise ValueError("query_state and prompt_summary aliases disagree")
        query_state = query_state if query_state is not None else prompt_summary
        return particle_mask, query_state

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        *,
        particle_z: Tensor | None = None,
        particle_mask: Tensor | None = None,
        response_mask: Tensor | None = None,
        query_state: Tensor | None = None,
        prompt_summary: Tensor | None = None,
        prompt_mask: Tensor | None = None,
        terminal_mask: Tensor | None = None,
        return_q: bool = True,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> ParticleModelOutput | tuple[Any, ...]:
        self._validate_runtime()
        particle_mask, query_state = self._resolve_aliases(
            particle_mask, response_mask, query_state, prompt_summary
        )
        if particle_z is not None and particle_mask is None:
            raise ValueError(
                "particle_mask/response_mask is required with particle_z; refusing to inject "
                "into prompt positions implicitly"
            )
        if particle_z is not None and query_state is None:
            raise ValueError(
                "query_state/prompt_summary is required for particle injection; obtain it from "
                "a prior clean prefill so the adapter is genuinely query-conditioned"
            )
        if not self._forward_lock.acquire(blocking=False):
            raise HrmIntegrationError(
                "re-entrant or concurrent forwards on one particle wrapper are unsupported"
            )

        hook_state: dict[str, Any] = {
            "calls": 0,
            "final_hidden": None,
            "adapter_output": None,
        }

        def high_cycle_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            call_index = hook_state["calls"]
            hook_state["calls"] += 1
            if not isinstance(output, Tensor):
                raise HrmIntegrationError(
                    "core.H_module no longer returns a tensor; upstream HRM integration changed"
                )
            should_inject = (
                call_index == self._capabilities.injection_after_high_cycle
                and particle_z is not None
            )
            if should_inject:
                adapter_output = self.adapter(
                    output,
                    query_state=query_state,
                    particle_z=particle_z,
                    response_mask=particle_mask,
                )
                hook_state["adapter_output"] = adapter_output
                output = adapter_output.hidden_states
            if call_index == self._capabilities.high_cycles - 1:
                hook_state["final_hidden"] = output
            return output

        handle = self._capabilities.high_module.register_forward_hook(high_cycle_hook)
        base_kwargs: dict[str, Any] = dict(kwargs)
        if input_ids is not None:
            base_kwargs["input_ids"] = input_ids
        if inputs_embeds is not None:
            base_kwargs["inputs_embeds"] = inputs_embeds
        if attention_mask is not None:
            base_kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            # PrefixLM semantics are data/condition specific; never infer this.
            base_kwargs["token_type_ids"] = token_type_ids
        if past_key_values is not None:
            base_kwargs["past_key_values"] = past_key_values
        if use_cache is not None:
            base_kwargs["use_cache"] = use_cache
        base_kwargs["return_dict"] = True

        try:
            base_output = self.base_model(**base_kwargs)
            if hook_state["calls"] != self._capabilities.high_cycles:
                raise HrmIntegrationError(
                    "unexpected H_module invocation count: expected "
                    f"{self._capabilities.high_cycles}, observed {hook_state['calls']}; "
                    "upstream HRM recurrence likely changed"
                )
            final_hidden = hook_state["final_hidden"]
            if final_hidden is None:
                raise HrmIntegrationError("failed to capture the final HRM high state")
        finally:
            handle.remove()
            self._forward_lock.release()

        logits = _get_output_field(base_output, "logits")
        if not isinstance(logits, Tensor):
            raise HrmIntegrationError(
                "base model output does not expose tensor logits; pass HrmTextForCausalLM, "
                "not the bare HrmTextModel"
            )

        # A clean prefill produces the query representation from the final H
        # state at the final prefix token.  An explicitly supplied state remains
        # fixed for every cached response step and particle branch.
        if query_state is None:
            summary_mask = prompt_mask
            if summary_mask is None and token_type_ids is not None:
                summary_mask = token_type_ids == 1
            if summary_mask is None:
                summary_mask = attention_mask
            query_state = _select_last_state(
                final_hidden,
                summary_mask,
                fallback_mask=attention_mask,
                name="prompt_mask",
            ).detach()
        else:
            if query_state.ndim == 3 and query_state.shape[1] == 1:
                query_state = query_state[:, 0]
            if query_state.ndim != 2 or query_state.shape != (
                final_hidden.shape[0],
                self._capabilities.hidden_size,
            ):
                raise ValueError(
                    "query_state must have shape "
                    f"[{final_hidden.shape[0]}, {self._capabilities.hidden_size}], "
                    f"got {tuple(query_state.shape)}"
                )

        if terminal_mask is None:
            terminal_mask = particle_mask
        terminal_state = _select_last_state(
            final_hidden,
            terminal_mask,
            fallback_mask=attention_mask,
            name="terminal_mask",
        )
        q_logits = None
        if return_q:
            q_terminal = terminal_state.detach() if self.detach_q_state else terminal_state
            q_query = query_state.detach() if self.detach_q_state else query_state
            q_logits = self.q_head(q_terminal, q_query)

        adapter_output: ParticleAdapterOutput | None = hook_state["adapter_output"]
        result = ParticleModelOutput(
            logits=logits,
            past_key_values=_get_output_field(base_output, "past_key_values"),
            terminal_state=terminal_state,
            prompt_summary=query_state,
            q_logits=q_logits,
            injection_delta=None if adapter_output is None else adapter_output.delta,
            relative_rms=None if adapter_output is None else adapter_output.relative_rms,
            injection_amplitude=None if adapter_output is None else adapter_output.amplitude,
            loss=_get_output_field(base_output, "loss"),
            hidden_states=_get_output_field(base_output, "hidden_states"),
            attentions=_get_output_field(base_output, "attentions"),
            base_output=base_output,
        )
        return result if return_dict else result.to_tuple()


# Concise alias used in configs and older notebook drafts.
HrmParticleModel = ParticleHrmForCausalLM


__all__ = [
    "HrmIntegrationError",
    "HrmParticleModel",
    "HrmTextCapabilities",
    "ParticleHrmForCausalLM",
    "ParticleModelOutput",
    "validate_hrm_text_model",
]
