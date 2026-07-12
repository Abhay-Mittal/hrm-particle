"""Parameter-free, direct Gaussian interventions in the HRM high state.

Unlike :class:`hrm_particle.adapter.ParticleAdapter`, this module does not map
a low-dimensional particle code through learned projections.  The particle is
itself a hidden-sized Gaussian direction.  It is RMS-normalized and added
directly to the current H state at a fixed fraction of that state's per-token
RMS.  Reusing one ``particle_z`` across decoding steps therefore implements a
coherent, response-level PTRM-style branch without adding trainable capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .adapter import ParticleAdapter, ParticleAdapterOutput, _rms, _zero_exact_safe_rms


@dataclass(frozen=True)
class GaussianParticleConfig:
    """Configuration for :class:`GaussianParticleAdapter`.

    ``relative_rms_scale`` is the intended ratio
    ``RMS(delta_h) / RMS(h)`` at every active H-state position.  Values are
    expressed as fractions, so ``0.03`` means a three-percent intervention.
    """

    hidden_size: int
    relative_rms_scale: float = 0.03
    rms_eps: float = 1e-6
    detach_reference_rms: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= self.relative_rms_scale <= 1.0:
            raise ValueError("relative_rms_scale must lie in [0, 1]")
        if self.rms_eps <= 0:
            raise ValueError("rms_eps must be positive")


class GaussianParticleAdapter(nn.Module):
    r"""Inject a normalized hidden-sized Gaussian direction directly into H.

    For hidden state :math:`h`, hidden-sized particle :math:`\epsilon`, and
    configured relative scale :math:`\alpha`, the active-position update is

    .. math::

        \Delta h = \alpha\,\operatorname{RMS}(h)
        \frac{\epsilon}{\max(\operatorname{RMS}(\epsilon), \varepsilon)}.

    There are no trainable parameters.  The selected scale is a persistent FP32
    buffer so it is covered by artifact checksums and can be changed during a
    predeclared development sweep. A zero particle remains bit-exactly zero,
    and a false response-mask position is also bit-exactly unchanged.
    ``query_state`` is accepted for drop-in compatibility with
    :class:`ParticleAdapter` but deliberately does not affect the direction.
    """

    def __init__(self, config: GaussianParticleConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "relative_rms_scale",
            torch.tensor(float(config.relative_rms_scale), dtype=torch.float32),
            persistent=True,
        )

    def set_relative_rms_scale(self, value: float) -> None:
        """Set a predeclared sweep/frozen scale without creating a parameter."""

        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("relative_rms_scale must lie in [0, 1]")
        self.relative_rms_scale.fill_(value)

    @property
    def latent_size(self) -> int:
        """The Gaussian particle is one full hidden-state direction."""

        return self.config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def _validate_query(self, query_state: Tensor, batch_size: int) -> None:
        if query_state.ndim == 3 and query_state.shape[1] == 1:
            query_state = query_state[:, 0]
        expected = (batch_size, self.hidden_size)
        if query_state.ndim != 2 or tuple(query_state.shape) != expected:
            raise ValueError(
                f"query_state must have shape [{batch_size}, {self.hidden_size}], "
                f"got {tuple(query_state.shape)}"
            )

    def _prepare_particle(self, particle_z: Tensor, batch_size: int, hidden_states: Tensor) -> Tensor:
        particle_z = torch.as_tensor(particle_z, device=hidden_states.device)
        if particle_z.ndim == 1:
            particle_z = particle_z.unsqueeze(0)
        if particle_z.ndim != 2 or particle_z.shape[-1] != self.hidden_size:
            raise ValueError(
                f"particle_z must have shape [batch, {self.hidden_size}], "
                f"got {tuple(particle_z.shape)}"
            )
        if particle_z.shape[0] == 1 and batch_size != 1:
            particle_z = particle_z.expand(batch_size, -1)
        elif particle_z.shape[0] != batch_size:
            raise ValueError(
                f"particle_z batch {particle_z.shape[0]} does not match hidden batch {batch_size}"
            )
        return particle_z

    def forward(
        self,
        hidden_states: Tensor,
        query_state: Tensor,
        particle_z: Tensor,
        response_mask: Tensor | None = None,
    ) -> ParticleAdapterOutput:
        if hidden_states.ndim < 2:
            raise ValueError(
                "hidden_states must have shape [batch, ..., hidden_size], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states last dimension must be {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )

        batch_size = hidden_states.shape[0]
        self._validate_query(query_state, batch_size)
        particle_z = self._prepare_particle(particle_z, batch_size, hidden_states)

        # Normalize in FP32 even when the model runs in BF16.  clamp_min keeps
        # the zero-particle path finite while preserving an exact zero numerator.
        particle_float = particle_z.float()
        particle_rms = particle_float.square().mean(dim=-1, keepdim=True).sqrt()
        unit_direction = particle_float / particle_rms.clamp_min(self.config.rms_eps)

        reference = hidden_states.detach() if self.config.detach_reference_rms else hidden_states
        reference_rms = reference.float().square().mean(dim=-1, keepdim=True).sqrt()

        # Broadcast one response-level direction over all token-like axes.  The
        # current token's own RMS determines magnitude, as in ParticleAdapter.
        for _ in range(hidden_states.ndim - 2):
            unit_direction = unit_direction.unsqueeze(1)
        scale = self.relative_rms_scale.float()
        delta = unit_direction * reference_rms * scale
        delta = delta.to(dtype=hidden_states.dtype)

        mask = ParticleAdapter._align_mask(response_mask, hidden_states)
        delta = delta * mask.unsqueeze(-1).to(dtype=delta.dtype)
        injected = hidden_states + delta

        relative_rms = _zero_exact_safe_rms(
            delta, eps=self.config.rms_eps
        ) / _rms(reference).float().clamp_min(self.config.rms_eps)
        relative_rms = relative_rms.to(dtype=hidden_states.dtype)

        amplitude_shape = (batch_size,) + (1,) * (hidden_states.ndim - 1)
        amplitude = torch.ones(
            amplitude_shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ) * scale.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return ParticleAdapterOutput(
            hidden_states=injected,
            delta=delta,
            relative_rms=relative_rms,
            amplitude=amplitude,
        )


# Descriptive alias for papers/configuration code that calls the operation an
# intervention instead of an adapter.  Both names denote the same zero-parameter
# implementation.
GaussianHStateIntervention = GaussianParticleAdapter


__all__ = [
    "GaussianHStateIntervention",
    "GaussianParticleAdapter",
    "GaussianParticleConfig",
]
