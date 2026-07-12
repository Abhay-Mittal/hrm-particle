"""Small trainable modules used by the HRM particle proof of concept.

The actor backbone is intentionally frozen.  :class:`ParticleAdapter` learns a
question-conditioned direction in the HRM high-state space and applies that
direction at a bounded fraction of the current state's RMS.  The construction
is residualized in the particle code, so ``z == 0`` produces an *exact* zero
tensor rather than merely a small one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _rms(value: Tensor, *, eps: float = 0.0) -> Tensor:
    """Return an RMS over the hidden dimension while retaining that dimension."""

    squared_mean = value.float().square().mean(dim=-1, keepdim=True)
    if eps:
        squared_mean = squared_mean + eps
    return squared_mean.sqrt().to(dtype=value.dtype)


def _zero_exact_safe_rms(value: Tensor, *, eps: float) -> Tensor:
    """Return an RMS that is exactly zero at zero with a finite derivative.

    ``sqrt(mean(x**2))`` has an infinite derivative at ``x == 0``.  In the
    particle adapter that matters even when a later mask selects only nonzero
    explorer positions: autograd can still encounter ``0 * inf`` through the
    exact-zero anchor and prompt positions and turn otherwise valid gradients
    into NaNs.  Subtracting the same positive floor keeps the reported value
    bit-exactly zero while making the derivative finite.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    squared_mean = value.float().square().mean(dim=-1, keepdim=True)
    # Squaring the configured RMS epsilon makes the floor negligible at the
    # intended 0.03--0.10 relative amplitudes while retaining stable backward.
    stabilizer = squared_mean.new_tensor(eps).square()
    floor = stabilizer.sqrt()
    return ((squared_mean + stabilizer).sqrt() - floor).to(dtype=value.dtype)


class RMSNormNoWeight(nn.Module):
    """Parameter-free RMS normalization.

    The computation is performed in float32 for stability and cast back to the
    input dtype, matching the normalization style used by HRM-Text.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = float(eps)

    def forward(self, value: Tensor) -> Tensor:
        scale = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (value.float() * scale).to(dtype=value.dtype)


@dataclass(frozen=True)
class ParticleAdapterConfig:
    """Configuration for :class:`ParticleAdapter`."""

    hidden_size: int
    latent_size: int = 64
    bottleneck_size: int = 64
    max_relative_rms: float = 0.10
    initial_relative_rms: float = 0.03
    rms_eps: float = 1e-6
    detach_query: bool = True
    detach_reference_rms: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.latent_size <= 0:
            raise ValueError("latent_size must be positive")
        if self.bottleneck_size <= 0:
            raise ValueError("bottleneck_size must be positive")
        if not 0.0 < self.max_relative_rms <= 1.0:
            raise ValueError("max_relative_rms must lie in (0, 1]")
        if not 0.0 < self.initial_relative_rms < self.max_relative_rms:
            raise ValueError("initial_relative_rms must lie in (0, max_relative_rms)")
        if self.rms_eps <= 0:
            raise ValueError("rms_eps must be positive")


@dataclass
class ParticleAdapterOutput:
    """Result of one particle injection."""

    hidden_states: Tensor
    delta: Tensor
    relative_rms: Tensor
    amplitude: Tensor


class ParticleAdapter(nn.Module):
    r"""Query-conditioned, response-level latent adapter.

    For a prompt summary :math:`c` and a response-level latent :math:`z`, the
    unnormalized direction is

    .. math::

        W_o[\operatorname{SiLU}(W_c c + W_z z)
            - \operatorname{SiLU}(W_c c)].

    Reusing the exact same projected ``c`` tensor on both sides makes the
    bracket bitwise zero when ``z`` is zero.  ``W_o`` is bias-free, so the final
    delta is also exactly zero.  A learned query-dependent gate sets the size,
    which is bounded by ``max_relative_rms`` times each H-state token's RMS.

    The same direction is broadcast over sequence positions; a response mask
    decides which positions receive it.  Reusing one ``z`` for every response
    step therefore gives a coherent sequence-level branch.
    """

    def __init__(self, config: ParticleAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.query_norm = RMSNormNoWeight(config.rms_eps)
        self.query_proj = nn.Linear(config.hidden_size, config.bottleneck_size, bias=True)
        self.latent_proj = nn.Linear(config.latent_size, config.bottleneck_size, bias=False)
        self.output_proj = nn.Linear(config.bottleneck_size, config.hidden_size, bias=False)
        self.amplitude_proj = nn.Linear(config.hidden_size, 1, bias=True)
        self.reset_parameters()

    @property
    def latent_size(self) -> int:
        return self.config.latent_size

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def reset_parameters(self) -> None:
        # Non-zero direction weights make the first rollout exploratory.  The
        # RMS normalization below controls magnitude independently of scale.
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.latent_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

        # Start every query at the requested relative amplitude while allowing
        # the gate to become query-dependent during training.
        nn.init.zeros_(self.amplitude_proj.weight)
        probability = self.config.initial_relative_rms / self.config.max_relative_rms
        logit = math.log(probability / (1.0 - probability))
        nn.init.constant_(self.amplitude_proj.bias, logit)

    def _prepare_batch_inputs(
        self,
        hidden_states: Tensor,
        query_state: Tensor,
        particle_z: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if hidden_states.ndim < 2:
            raise ValueError(
                "hidden_states must have shape [batch, ..., hidden_size], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch_size = hidden_states.shape[0]
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states last dimension must be {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )

        if query_state.ndim == 3 and query_state.shape[1] == 1:
            query_state = query_state[:, 0]
        if query_state.ndim != 2 or query_state.shape != (batch_size, self.hidden_size):
            raise ValueError(
                "query_state must have shape "
                f"[{batch_size}, {self.hidden_size}], got {tuple(query_state.shape)}"
            )

        if particle_z.ndim == 1:
            particle_z = particle_z.unsqueeze(0)
        if particle_z.ndim != 2 or particle_z.shape[-1] != self.latent_size:
            raise ValueError(
                f"particle_z must have shape [batch, {self.latent_size}], "
                f"got {tuple(particle_z.shape)}"
            )
        if particle_z.shape[0] == 1 and batch_size != 1:
            particle_z = particle_z.expand(batch_size, -1)
        elif particle_z.shape[0] != batch_size:
            raise ValueError(
                f"particle_z batch {particle_z.shape[0]} does not match hidden batch {batch_size}"
            )

        parameter_dtype = self.query_proj.weight.dtype
        parameter_device = self.query_proj.weight.device
        query_state = query_state.to(device=parameter_device, dtype=parameter_dtype)
        particle_z = particle_z.to(device=parameter_device, dtype=parameter_dtype)
        return query_state, particle_z

    @staticmethod
    def _align_mask(mask: Tensor | None, hidden_states: Tensor) -> Tensor:
        leading_shape = hidden_states.shape[:-1]
        if mask is None:
            return torch.ones(leading_shape, dtype=torch.bool, device=hidden_states.device)

        mask = torch.as_tensor(mask, device=hidden_states.device)
        if hidden_states.ndim == 2:
            if mask.ndim == 2 and mask.shape[1] == 1:
                mask = mask[:, 0]
            if mask.ndim != 1 or mask.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    f"mask must have shape [{hidden_states.shape[0]}], got {tuple(mask.shape)}"
                )
            return mask.bool()

        if mask.ndim != hidden_states.ndim - 1:
            raise ValueError(
                f"mask must have {hidden_states.ndim - 1} dimensions, got {mask.ndim}"
            )
        if mask.shape[0] != hidden_states.shape[0]:
            raise ValueError("mask and hidden_states batch dimensions do not match")
        # During cached decoding callers sometimes retain a full-history mask
        # while the model receives only the unprocessed suffix.
        for dimension in range(1, mask.ndim):
            target = leading_shape[dimension]
            current = mask.shape[dimension]
            if current < target:
                raise ValueError(
                    f"mask dimension {dimension} has length {current}, expected at least {target}"
                )
            if current > target:
                index = [slice(None)] * mask.ndim
                index[dimension] = slice(current - target, None)
                mask = mask[tuple(index)]
        if tuple(mask.shape) != tuple(leading_shape):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} cannot align to {tuple(leading_shape)}"
            )
        return mask.bool()

    def forward(
        self,
        hidden_states: Tensor,
        query_state: Tensor,
        particle_z: Tensor,
        response_mask: Tensor | None = None,
    ) -> ParticleAdapterOutput:
        query_state, particle_z = self._prepare_batch_inputs(
            hidden_states, query_state, particle_z
        )
        if self.config.detach_query:
            query_state = query_state.detach()

        normalized_query = self.query_norm(query_state)
        conditioned = self.query_proj(normalized_query)
        latent_offset = self.latent_proj(particle_z)

        # This exact residualization is the anchor guarantee: latent_offset is
        # exactly zero at z=0, the two activations are identical, and all later
        # bias-free operations preserve zero exactly.
        direction_features = F.silu(conditioned + latent_offset) - F.silu(conditioned)
        raw_direction = self.output_proj(direction_features)
        direction_rms = _rms(raw_direction, eps=self.config.rms_eps)
        unit_direction = raw_direction / direction_rms

        amplitude = self.config.max_relative_rms * torch.sigmoid(
            self.amplitude_proj(normalized_query)
        )

        reference = hidden_states.detach() if self.config.detach_reference_rms else hidden_states
        reference_rms = _rms(reference)

        # Expand [B, H] and [B, 1] across all token-like leading dimensions.
        expand_dimensions = hidden_states.ndim - 2
        for _ in range(expand_dimensions):
            unit_direction = unit_direction.unsqueeze(1)
            amplitude = amplitude.unsqueeze(1)
        delta = unit_direction.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        ) * reference_rms * amplitude.to(device=hidden_states.device, dtype=hidden_states.dtype)

        mask = self._align_mask(response_mask, hidden_states)
        delta = delta * mask.unsqueeze(-1).to(delta.dtype)

        # Adding an exact floating-point zero is bit-preserving, which is
        # verified by the offline tests for both z=0 and masked prompt tokens.
        injected = hidden_states + delta
        # This diagnostic is used by the differentiable injection penalty.
        # It must remain safe at the many exact-zero prompt/anchor positions.
        relative_rms = _zero_exact_safe_rms(
            delta, eps=self.config.rms_eps
        ) / reference_rms.float().clamp_min(self.config.rms_eps)
        relative_rms = relative_rms.to(dtype=hidden_states.dtype)
        return ParticleAdapterOutput(
            hidden_states=injected,
            delta=delta,
            relative_rms=relative_rms,
            amplitude=amplitude,
        )


class SharedQHead(nn.Module):
    """A small query-aware, shared terminal-state correctness head.

    The same head is evaluated for every anchor/particle candidate.  Prompt and
    terminal states receive separate parameter-free RMS normalization and
    projections before being combined.  It returns logits (not probabilities),
    suitable for ``binary_cross_entropy_with_logits``.

    ``query_state`` is optional only for terminal-only ablations and backwards
    compatibility.  The end-to-end particle experiment should always provide it.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int | None = None) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        bottleneck_size = bottleneck_size or max(32, hidden_size // 4)
        if bottleneck_size <= 0:
            raise ValueError("bottleneck_size must be positive")
        self.hidden_size = hidden_size
        self.terminal_norm = RMSNormNoWeight()
        self.query_norm = RMSNormNoWeight()
        self.terminal_proj = nn.Linear(hidden_size, bottleneck_size)
        self.query_proj = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.output_proj = nn.Linear(bottleneck_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.terminal_proj.weight)
        nn.init.zeros_(self.terminal_proj.bias)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def initialize_constant_prior(self, probability: float) -> None:
        """Make every candidate start at the same correctness prior.

        Only the final projection is changed.  Its zero weight makes the head
        independent of the randomly initialized hidden features, while the
        bias is the logit of ``probability``.  Consequently, an ``argmax``
        selector deterministically falls back to branch zero until Q learns a
        candidate-dependent score.
        """

        probability = float(probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie strictly between 0 and 1")
        prior_logit = math.log(probability) - math.log1p(-probability)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.fill_(prior_logit)

    def forward(self, terminal_state: Tensor, query_state: Tensor | None = None) -> Tensor:
        if terminal_state.ndim != 2 or terminal_state.shape[-1] != self.hidden_size:
            raise ValueError(
                f"terminal_state must have shape [batch, {self.hidden_size}], "
                f"got {tuple(terminal_state.shape)}"
            )
        if query_state is None:
            query_state = torch.zeros_like(terminal_state)
        if query_state.shape != terminal_state.shape:
            raise ValueError(
                "query_state must have the same [batch, hidden] shape as terminal_state, "
                f"got {tuple(query_state.shape)} and {tuple(terminal_state.shape)}"
            )
        parameter = self.terminal_proj.weight
        terminal_state = terminal_state.to(device=parameter.device, dtype=parameter.dtype)
        query_state = query_state.to(device=parameter.device, dtype=parameter.dtype)
        terminal_features = self.terminal_proj(self.terminal_norm(terminal_state))
        query_features = self.query_proj(self.query_norm(query_state))
        hidden = F.silu(terminal_features + query_features)
        return self.output_proj(hidden).squeeze(-1)


# Descriptive aliases retained for call sites and experiment configs.
QueryConditionedParticleAdapter = ParticleAdapter
TerminalQHead = SharedQHead


__all__ = [
    "ParticleAdapter",
    "ParticleAdapterConfig",
    "ParticleAdapterOutput",
    "QueryConditionedParticleAdapter",
    "RMSNormNoWeight",
    "SharedQHead",
    "TerminalQHead",
]
