"""Small safety and runtime utilities used by the V1 training notebook.

The functions in this module deliberately have no dataset, network, or model
dependencies.  That keeps the selector gate and checkpoint path independently
testable before an expensive multi-GPU run.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn


_FENCE_RE = re.compile(
    r"(?ms)^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n]*)\n"
    r"(?P<body>.*?)^[ \t]*(?P=fence)[ \t]*$"
)
_PYTHON_FENCE_NAMES = frozenset({"py", "python", "python3"})
Q_SPLIT_NAMES = ("train", "early_stop", "margin_select", "safety_test")
Q_SPLIT_ALGORITHM = "source-stratified-sha256-largest-remainder-v1"


def deterministic_prompt_split(
    identifier: str,
    percentages: Mapping[str, int],
) -> str:
    """Assign one prompt to a stable, mutually exclusive Q-data split.

    The split is derived only from the full prompt identifier, so candidates
    from one K-way rollout can never leak across train/calibration partitions.
    Percentages are integer points out of 100 to keep the assignment and its
    provenance independent of floating-point parsing.
    """

    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier must be a non-empty string")
    if set(percentages) != set(Q_SPLIT_NAMES):
        raise ValueError("Q split percentages must define " + ", ".join(Q_SPLIT_NAMES))
    clean: dict[str, int] = {}
    for name in Q_SPLIT_NAMES:
        value = percentages[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Q split percentage {name!r} must be a positive integer")
        clean[name] = value
    if sum(clean.values()) != 100:
        raise ValueError("Q split percentages must sum to 100")

    bucket = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    upper = 0
    for name in Q_SPLIT_NAMES:
        upper += clean[name] * 100
        if bucket < upper:
            return name
    raise AssertionError("a 100-percent partition must contain every hash bucket")


def prompt_split_masks(
    identifiers: Sequence[str],
    percentages: Mapping[str, int],
) -> dict[str, Tensor]:
    """Return boolean prompt-group masks for all four disjoint Q splits."""

    if not identifiers:
        raise ValueError("identifiers must not be empty")
    assignments = [deterministic_prompt_split(value, percentages) for value in identifiers]
    masks = {
        name: torch.tensor([value == name for value in assignments], dtype=torch.bool)
        for name in Q_SPLIT_NAMES
    }
    stacked = torch.stack(list(masks.values()))
    if not bool(stacked.sum(dim=0).eq(1).all()):
        raise AssertionError("Q prompt splits must be mutually exclusive and exhaustive")
    return masks


def stratified_prompt_splits(
    identifiers: Sequence[str],
    strata: Sequence[str],
    percentages: Mapping[str, int],
) -> list[str]:
    """Assign exact, deterministic split quotas independently within each stratum."""

    if not identifiers or len(identifiers) != len(strata):
        raise ValueError("identifiers and strata must be non-empty and aligned")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("prompt identifiers must be unique")
    # Reuse the strict percentage validation without making hash buckets the
    # actual assignment mechanism.
    deterministic_prompt_split(identifiers[0], percentages)
    by_stratum: dict[str, list[int]] = {}
    for index, raw_stratum in enumerate(strata):
        stratum = str(raw_stratum)
        if not stratum:
            raise ValueError("strata must be non-empty strings")
        by_stratum.setdefault(stratum, []).append(index)

    assignments = [""] * len(identifiers)
    for indices in by_stratum.values():
        indices.sort(
            key=lambda index: hashlib.sha256(
                identifiers[index].encode("utf-8")
            ).digest()
        )
        count = len(indices)
        quotas = {
            name: count * int(percentages[name]) // 100 for name in Q_SPLIT_NAMES
        }
        remaining = count - sum(quotas.values())
        fractional_order = sorted(
            Q_SPLIT_NAMES,
            key=lambda name: (
                -(count * int(percentages[name]) % 100),
                Q_SPLIT_NAMES.index(name),
            ),
        )
        for name in fractional_order[:remaining]:
            quotas[name] += 1
        cursor = 0
        for name in Q_SPLIT_NAMES:
            for index in indices[cursor : cursor + quotas[name]]:
                assignments[index] = name
            cursor += quotas[name]
        if cursor != count:
            raise AssertionError("stratified Q split quotas must be exhaustive")
    if any(not value for value in assignments):
        raise AssertionError("every prompt must receive a Q split")
    return assignments


def stratified_prompt_split_masks(
    identifiers: Sequence[str],
    strata: Sequence[str],
    percentages: Mapping[str, int],
) -> dict[str, Tensor]:
    assignments = stratified_prompt_splits(identifiers, strata, percentages)
    return {
        name: torch.tensor([value == name for value in assignments], dtype=torch.bool)
        for name in Q_SPLIT_NAMES
    }


def extract_python_code(text: str) -> str:
    """Extract the most plausible Python program from a model response.

    Python-labelled Markdown fences are preferred over unlabelled fences.  If
    there is more than one eligible block, the longest non-whitespace block is
    selected (with an earliest-block tie break).  This handles the common case
    where a short illustrative snippet precedes the complete answer.  Raw text
    is returned unchanged apart from surrounding whitespace when no eligible
    fence is present.

    A fence labelled with another language is never silently executed as
    Python.  If a response consists only of such fences, the empty string is
    returned so the verifier can mark it unparseable.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        return text.strip()

    python_blocks: list[tuple[int, str]] = []
    unlabelled_blocks: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        # Only the first info-string token denotes the language.  Attributes
        # after it (for example ``python linenums``) do not change the choice.
        info = match.group("info").strip().lower()
        language = info.split(maxsplit=1)[0] if info else ""
        body = match.group("body").strip()
        if language in _PYTHON_FENCE_NAMES:
            python_blocks.append((index, body))
        elif not language:
            unlabelled_blocks.append((index, body))

    candidates = python_blocks or unlabelled_blocks
    if not candidates:
        return ""
    # ``max`` keeps the first candidate on a length tie because the negated
    # source index is the secondary key.
    _, body = max(candidates, key=lambda item: (len(item[1].strip()), -item[0]))
    return body


def deterministic_math_majority_vote(
    normalized_answers: Sequence[Optional[str]],
) -> int:
    """Return the selected sample index for already-normalized math answers.

    ``None`` and the empty string are treated as unparseable and do not receive
    votes.  Ties are resolved by the earliest occurrence in the input.  If all
    samples are unparseable, index zero is returned as a conservative fallback.

    Normalization is intentionally the caller's responsibility: answer
    extraction and mathematical equivalence are benchmark-specific, whereas
    voting itself should be deterministic and auditable.
    """

    if isinstance(normalized_answers, (str, bytes)):
        raise TypeError("normalized_answers must be a sequence, not a string")
    if not normalized_answers:
        raise ValueError("normalized_answers must not be empty")

    counts: dict[str, int] = {}
    for answer in normalized_answers:
        if answer is not None and not isinstance(answer, str):
            raise TypeError("each normalized answer must be a string or None")
        if answer not in (None, ""):
            counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return 0

    largest_count = max(counts.values())
    for index, answer in enumerate(normalized_answers):
        if answer not in (None, "") and counts[answer] == largest_count:
            return index
    raise AssertionError("a non-empty vote table must have a winner")


# Short aliases are convenient in evaluation cells while retaining an explicit
# canonical name in logs and tests.
math_majority_vote = deterministic_math_majority_vote
majority_vote_normalized = deterministic_math_majority_vote


def _as_group_matrix(values: Tensor, *, name: str, min_candidates: int = 1) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [num_prompts, K]")
    if values.shape[0] == 0 or values.shape[1] < min_candidates:
        raise ValueError(
            f"{name} must contain at least one prompt and {min_candidates} candidate(s)"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validate_binary_correctness(correctness: Tensor) -> Tensor:
    correctness = _as_group_matrix(correctness, name="correctness")
    if not bool(((correctness == 0) | (correctness == 1)).all()):
        raise ValueError("correctness must contain only binary values 0 or 1")
    return correctness


def select_with_anchor_fallback(
    logits: Tensor,
    margin: float,
    ready: bool | Tensor,
) -> Tensor:
    """Select an explorer only when a ready Q head clears the anchor margin.

    Branch zero is the unmodified anchor.  For every prompt the best explorer
    is selected only when its logit is *strictly greater* than
    ``anchor_logit + margin``.  Equality therefore falls back to the anchor.
    ``ready`` can be one global boolean or a boolean mask of length ``batch``.
    """

    logits = _as_group_matrix(logits, name="logits")
    if not logits.dtype.is_floating_point:
        raise TypeError("logits must have a floating-point dtype")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise TypeError("margin must be a real scalar")
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")

    batch, candidates = logits.shape
    if isinstance(ready, bool):
        ready_mask = torch.full((batch,), ready, dtype=torch.bool, device=logits.device)
    elif isinstance(ready, Tensor):
        if ready.dtype != torch.bool:
            raise TypeError("a tensor ready mask must have dtype torch.bool")
        if ready.ndim != 1 or ready.shape[0] != batch:
            raise ValueError("a tensor ready mask must have shape [num_prompts]")
        ready_mask = ready.to(device=logits.device)
    else:
        raise TypeError("ready must be a bool or boolean tensor")

    selected = torch.zeros(batch, dtype=torch.long, device=logits.device)
    if candidates == 1 or not bool(ready_mask.any()):
        return selected
    explorer_logits, relative_indices = logits[:, 1:].max(dim=1)
    clears_margin = explorer_logits > (logits[:, 0] + margin)
    use_explorer = ready_mask & clears_margin
    selected[use_explorer] = relative_indices[use_explorer] + 1
    return selected


@dataclass(frozen=True)
class MarginTrial:
    """Held-out selector statistics for one candidate margin."""

    margin: float
    selected_accuracy: float
    mean_delta: float
    ci_low: float
    ci_high: float
    switch_count: int
    switch_fraction: float
    passes_no_degradation_gate: bool


@dataclass(frozen=True)
class MarginCalibrationResult:
    """Auditable result of paired-bootstrap Q-margin calibration."""

    margin: float
    ready: bool
    reason: str
    num_prompts: int
    anchor_accuracy: float
    selected_accuracy: float
    mean_delta: float
    ci_low: float
    ci_high: float
    confidence: float
    switch_count: int
    switch_fraction: float
    max_degradation: float
    trials: tuple[MarginTrial, ...]


def _paired_bootstrap_interval(
    paired_deltas: Tensor,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a prompt-paired mean interval without a NumPy dependency."""

    values = paired_deltas.detach().to(device="cpu", dtype=torch.float64).flatten()
    if values.numel() == 0:
        raise ValueError("paired_deltas must not be empty")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    means: list[Tensor] = []
    remaining = samples
    # Bound temporary storage for large calibration sets.
    max_indices_per_chunk = 2_000_000
    chunk_size = max(1, min(samples, max_indices_per_chunk // values.numel()))
    while remaining:
        current = min(remaining, chunk_size)
        indices = torch.randint(
            values.numel(),
            (current, values.numel()),
            generator=generator,
        )
        means.append(values[indices].mean(dim=1))
        remaining -= current
    bootstrap_means = torch.cat(means)
    tail = (1.0 - confidence) / 2.0
    low, high = torch.quantile(
        bootstrap_means,
        torch.tensor([tail, 1.0 - tail], dtype=bootstrap_means.dtype),
    )
    return float(low), float(high)


def calibrate_q_margin(
    logits: Tensor,
    correctness: Tensor,
    *,
    margins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
    bootstrap_samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
    max_degradation: float = 0.0,
    min_prompts: int = 64,
    min_switches: int = 1,
) -> MarginCalibrationResult:
    """Choose a selector margin behind a paired no-degradation gate.

    Each candidate is evaluated on the same prompt groups against branch zero.
    A margin is eligible only when the lower bound of the paired-bootstrap
    confidence interval is at least ``-max_degradation`` and it switches at
    least ``min_switches`` prompts.  Among eligible margins, observed selected
    accuracy is maximized; ties prefer the larger, more conservative margin.

    ``ready=False`` means callers must use branch zero regardless of the
    returned diagnostic margin.  Calibration data must be held out from Q
    training; this function cannot enforce that experimental separation.
    """

    logits = _as_group_matrix(logits, name="logits", min_candidates=2)
    if not logits.dtype.is_floating_point:
        raise TypeError("logits must have a floating-point dtype")
    correctness = _validate_binary_correctness(correctness)
    if correctness.shape != logits.shape:
        raise ValueError("correctness must have the same shape as logits")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise TypeError("bootstrap_samples must be an integer")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if not isinstance(confidence, (int, float)) or not 0.5 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between 0.5 and 1")
    confidence = float(confidence)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(max_degradation, (int, float)):
        raise TypeError("max_degradation must be a real scalar")
    max_degradation = float(max_degradation)
    if not math.isfinite(max_degradation) or max_degradation < 0:
        raise ValueError("max_degradation must be finite and non-negative")
    for name, value in (("min_prompts", min_prompts), ("min_switches", min_switches)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    if isinstance(margins, (str, bytes)) or not margins:
        raise ValueError("margins must contain at least one candidate")
    clean_margins: list[float] = []
    for raw_margin in margins:
        if isinstance(raw_margin, bool) or not isinstance(raw_margin, (int, float)):
            raise TypeError("every margin must be a real scalar")
        candidate = float(raw_margin)
        if not math.isfinite(candidate) or candidate < 0:
            raise ValueError("every margin must be finite and non-negative")
        if candidate not in clean_margins:
            clean_margins.append(candidate)
    clean_margins.sort()

    logits_cpu = logits.detach().to(device="cpu", dtype=torch.float64)
    labels_cpu = correctness.detach().to(device="cpu", dtype=torch.float64)
    anchor = labels_cpu[:, 0]
    anchor_accuracy = float(anchor.mean())
    trials: list[MarginTrial] = []
    for trial_index, margin in enumerate(clean_margins):
        selected_indices = select_with_anchor_fallback(logits_cpu, margin, True)
        selected = labels_cpu.gather(1, selected_indices[:, None]).squeeze(1)
        deltas = selected - anchor
        low, high = _paired_bootstrap_interval(
            deltas,
            samples=bootstrap_samples,
            confidence=confidence,
            # Distinct deterministic streams avoid accidental identical index
            # draws while keeping an entire calibration reproducible.
            seed=seed + trial_index * 1_000_003,
        )
        switch_count = int((selected_indices != 0).sum())
        passes = (
            logits.shape[0] >= min_prompts
            and switch_count >= min_switches
            and low >= -max_degradation
        )
        trials.append(
            MarginTrial(
                margin=margin,
                selected_accuracy=float(selected.mean()),
                mean_delta=float(deltas.mean()),
                ci_low=low,
                ci_high=high,
                switch_count=switch_count,
                switch_fraction=switch_count / logits.shape[0],
                passes_no_degradation_gate=passes,
            )
        )

    eligible = [trial for trial in trials if trial.passes_no_degradation_gate]
    if eligible:
        # Conservatism is the last tie-break: a larger margin changes fewer
        # baseline decisions when held-out accuracies are indistinguishable.
        chosen = max(
            eligible,
            key=lambda trial: (trial.selected_accuracy, trial.mean_delta, trial.margin),
        )
        ready = True
        reason = "paired bootstrap no-degradation gate passed"
    else:
        # The most conservative attempted margin is the most useful diagnostic,
        # but ``ready=False`` makes the effective policy exactly branch zero.
        chosen = max(trials, key=lambda trial: trial.margin)
        ready = False
        if logits.shape[0] < min_prompts:
            reason = f"need at least {min_prompts} held-out prompt groups"
        elif max(trial.switch_count for trial in trials) < min_switches:
            reason = f"need at least {min_switches} explorer selections"
        else:
            reason = "no margin passed the paired bootstrap no-degradation gate"

    return MarginCalibrationResult(
        margin=chosen.margin,
        ready=ready,
        reason=reason,
        num_prompts=logits.shape[0],
        anchor_accuracy=anchor_accuracy,
        selected_accuracy=chosen.selected_accuracy,
        mean_delta=chosen.mean_delta,
        ci_low=chosen.ci_low,
        ci_high=chosen.ci_high,
        confidence=confidence,
        switch_count=chosen.switch_count,
        switch_fraction=chosen.switch_fraction,
        max_degradation=max_degradation,
        trials=tuple(trials),
    )


# A descriptive alias keeps notebook prose readable.
calibrate_margin = calibrate_q_margin


@dataclass(frozen=True)
class CodeSignalGateResult:
    """Whether a recent code-rollout window contains usable verifier signal."""

    enabled: bool
    reason: str
    num_groups: int
    nonzero_groups: int
    mixed_groups: int
    nonzero_fraction: float
    mixed_fraction: float


def code_signal_gate(
    correctness: Tensor,
    *,
    min_groups: int = 32,
    min_nonzero_groups: int = 4,
    min_mixed_groups: int = 4,
    min_nonzero_fraction: float = 0.05,
    min_mixed_fraction: float = 0.05,
) -> CodeSignalGateResult:
    """Disable code RL when strict test outcomes provide too little signal.

    A nonzero group has at least one fully correct candidate.  A mixed group has
    both correct and incorrect candidates and is especially important for
    within-prompt Q ranking.  The gate requires both absolute counts and
    fractions so it behaves sensibly for either a smoke-test window or a long
    production window.
    """

    correctness = _validate_binary_correctness(correctness)
    integer_limits = {
        "min_groups": min_groups,
        "min_nonzero_groups": min_nonzero_groups,
        "min_mixed_groups": min_mixed_groups,
    }
    for name, value in integer_limits.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    fraction_limits = {
        "min_nonzero_fraction": min_nonzero_fraction,
        "min_mixed_fraction": min_mixed_fraction,
    }
    for name, value in fraction_limits.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")

    group_sums = correctness.detach().to(device="cpu").sum(dim=1)
    num_groups, k = correctness.shape
    nonzero_groups = int((group_sums > 0).sum())
    mixed_groups = int(((group_sums > 0) & (group_sums < k)).sum())
    nonzero_fraction = nonzero_groups / num_groups
    mixed_fraction = mixed_groups / num_groups

    failures: list[str] = []
    if num_groups < min_groups:
        failures.append(f"groups {num_groups} < {min_groups}")
    if nonzero_groups < min_nonzero_groups:
        failures.append(f"nonzero groups {nonzero_groups} < {min_nonzero_groups}")
    if mixed_groups < min_mixed_groups:
        failures.append(f"mixed groups {mixed_groups} < {min_mixed_groups}")
    if nonzero_fraction < float(min_nonzero_fraction):
        failures.append(
            f"nonzero fraction {nonzero_fraction:.3f} < {float(min_nonzero_fraction):.3f}"
        )
    if mixed_fraction < float(min_mixed_fraction):
        failures.append(
            f"mixed fraction {mixed_fraction:.3f} < {float(min_mixed_fraction):.3f}"
        )
    enabled = not failures
    return CodeSignalGateResult(
        enabled=enabled,
        reason="signal gate passed" if enabled else "; ".join(failures),
        num_groups=num_groups,
        nonzero_groups=nonzero_groups,
        mixed_groups=mixed_groups,
        nonzero_fraction=nonzero_fraction,
        mixed_fraction=mixed_fraction,
    )


check_code_signal = code_signal_gate


class BF16MasterAdamW(torch.optim.Optimizer):
    """AdamW for BF16 model parameters with FP32 master weights and state.

    The public ``param_groups`` contain the original model parameters, which is
    important for gradient clipping and adapter-scope validation.  A private
    AdamW instance updates FP32 clones.  On every step BF16 gradients are copied
    to those clones, the FP32 update is performed, and rounded weights are
    copied back to the model.

    This is mixed-precision training in the robust sense: model weights and
    activations can remain BF16 while numerically sensitive optimizer state is
    FP32.  Sparse gradients are deliberately rejected because AdamW does not
    support them.
    """

    _STATE_FORMAT = "bf16-master-adamw"
    _STATE_VERSION = 1

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[Mapping[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
        }
        super().__init__(params, defaults)
        if not self.param_groups:
            raise ValueError("optimizer got an empty parameter list")

        seen: set[int] = set()
        master_groups: list[dict[str, Any]] = []
        self._model_master_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for group in self.param_groups:
            master_group = {key: value for key, value in group.items() if key != "params"}
            masters: list[nn.Parameter] = []
            for parameter in group["params"]:
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError("optimizer parameters must be nn.Parameter instances")
                if id(parameter) in seen:
                    raise ValueError("a parameter appears in more than one optimizer group")
                seen.add(id(parameter))
                if parameter.dtype != torch.bfloat16:
                    raise TypeError(
                        "BF16MasterAdamW requires torch.bfloat16 model parameters; "
                        f"got {parameter.dtype}"
                    )
                if not parameter.is_floating_point():
                    raise TypeError("optimizer parameters must be floating point")
                master = nn.Parameter(
                    parameter.detach().to(dtype=torch.float32).clone(),
                    requires_grad=True,
                )
                masters.append(master)
                self._model_master_pairs.append((parameter, master))
            master_group["params"] = masters
            master_groups.append(master_group)
        if not self._model_master_pairs:
            raise ValueError("optimizer got an empty parameter list")
        self._master_optimizer = torch.optim.AdamW(master_groups)

    @property
    def master_parameters(self) -> tuple[nn.Parameter, ...]:
        """The FP32 weights, exposed read-only for diagnostics."""

        return tuple(master for _, master in self._model_master_pairs)

    @property
    def master_optimizer(self) -> torch.optim.AdamW:
        """Underlying FP32 optimizer, exposed for state-dtype diagnostics."""

        return self._master_optimizer

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Reject late groups because they would lack corresponding masters."""

        # ``Optimizer.__init__`` dispatches through this method before the
        # internal optimizer exists, so initial parameter groups still work.
        if hasattr(self, "_master_optimizer"):
            raise RuntimeError(
                "BF16MasterAdamW parameter groups are fixed at construction time"
            )
        super().add_param_group(param_group)

    @torch.no_grad()
    def _copy_gradients_to_master(self) -> None:
        for model_parameter, master_parameter in self._model_master_pairs:
            gradient = model_parameter.grad
            if gradient is None:
                master_parameter.grad = None
                continue
            if gradient.is_sparse:
                raise RuntimeError("BF16MasterAdamW does not support sparse gradients")
            if gradient.shape != model_parameter.shape:
                raise RuntimeError("gradient shape does not match its model parameter")
            converted = gradient.detach().to(
                device=master_parameter.device,
                dtype=torch.float32,
            )
            if master_parameter.grad is None:
                master_parameter.grad = converted.clone()
            else:
                master_parameter.grad.copy_(converted)

    @torch.no_grad()
    def _copy_master_to_model(self) -> None:
        for model_parameter, master_parameter in self._model_master_pairs:
            model_parameter.copy_(master_parameter.to(dtype=torch.bfloat16))

    def _sync_public_hyperparameters(self) -> None:
        """Propagate scheduler edits from public groups to FP32 AdamW groups."""

        if len(self.param_groups) != len(self._master_optimizer.param_groups):
            raise RuntimeError("optimizer parameter-group count changed unexpectedly")
        for public, internal in zip(self.param_groups, self._master_optimizer.param_groups):
            for key, value in public.items():
                if key != "params":
                    internal[key] = value

    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:
        """Perform one FP32-master AdamW update and refresh BF16 weights."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._copy_gradients_to_master()
        self._sync_public_hyperparameters()
        self._master_optimizer.step()
        self._copy_master_to_model()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        # ``Optimizer.zero_grad`` acts on the public BF16 parameter groups.
        super().zero_grad(set_to_none=set_to_none)
        self._master_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return master weights plus the complete FP32 AdamW resume state."""

        self._sync_public_hyperparameters()
        return {
            "format": self._STATE_FORMAT,
            "version": self._STATE_VERSION,
            "master_weights": [
                parameter.detach().clone() for parameter in self.master_parameters
            ],
            # PyTorch optimizer state dictionaries can hold live tensor
            # references.  Clone them so an in-memory save/load does not couple
            # the resumed optimizer to a still-running source optimizer.
            "optimizer": copy.deepcopy(self._master_optimizer.state_dict()),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore master weights/state and synchronize the BF16 model copy."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("optimizer state_dict must be a mapping")
        if state_dict.get("format") != self._STATE_FORMAT:
            raise ValueError("not a BF16MasterAdamW state dictionary")
        if state_dict.get("version") != self._STATE_VERSION:
            raise ValueError(
                f"unsupported BF16MasterAdamW state version {state_dict.get('version')!r}"
            )
        weights = state_dict.get("master_weights")
        if not isinstance(weights, Sequence) or len(weights) != len(self.master_parameters):
            raise ValueError("master weight count does not match this optimizer")
        with torch.no_grad():
            for index, (saved, master) in enumerate(zip(weights, self.master_parameters)):
                if not isinstance(saved, Tensor):
                    raise TypeError(f"master weight {index} is not a tensor")
                if saved.shape != master.shape:
                    raise ValueError(
                        f"master weight {index} has shape {tuple(saved.shape)}, "
                        f"expected {tuple(master.shape)}"
                    )
                if not saved.dtype.is_floating_point:
                    raise TypeError(f"master weight {index} must be floating point")
                master.copy_(saved.to(device=master.device, dtype=torch.float32))
        optimizer_state = state_dict.get("optimizer")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("state dictionary contains no optimizer mapping")
        self._master_optimizer.load_state_dict(copy.deepcopy(dict(optimizer_state)))

        # Hyperparameters such as a scheduler-adjusted learning rate live in the
        # internal groups.  Mirror them to public groups without replacing the
        # BF16 model-parameter lists.
        if len(self.param_groups) != len(self._master_optimizer.param_groups):
            raise ValueError("optimizer parameter-group count changed during load")
        for public, internal in zip(self.param_groups, self._master_optimizer.param_groups):
            public_params = public["params"]
            public.clear()
            public.update({key: value for key, value in internal.items() if key != "params"})
            public["params"] = public_params
        self._copy_master_to_model()


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> Path:
    """Durably write a Torch payload and atomically replace ``path``.

    The temporary file is created in the destination directory, making
    ``os.replace`` atomic on ordinary local filesystems.  A failed serialization
    leaves an existing destination untouched and removes the partial file.
    """

    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.tmp-",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        # Persist the directory entry where the platform permits directory
        # fsync.  Windows and a few virtual filesystems may reject this step.
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BF16MasterAdamW",
    "CodeSignalGateResult",
    "MarginCalibrationResult",
    "MarginTrial",
    "atomic_torch_save",
    "calibrate_margin",
    "calibrate_q_margin",
    "check_code_signal",
    "code_signal_gate",
    "deterministic_math_majority_vote",
    "extract_python_code",
    "majority_vote_normalized",
    "math_majority_vote",
    "select_with_anchor_fallback",
    "sha256_file",
]
