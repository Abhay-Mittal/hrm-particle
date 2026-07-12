from __future__ import annotations

import pytest
import torch

from hrm_particle.adapter import ParticleAdapterOutput
from hrm_particle.gaussian import GaussianParticleAdapter, GaussianParticleConfig


def make_intervention(**overrides: object) -> GaussianParticleAdapter:
    values: dict[str, object] = {
        "hidden_size": 16,
        "relative_rms_scale": 0.03,
        "rms_eps": 1e-6,
    }
    values.update(overrides)
    return GaussianParticleAdapter(GaussianParticleConfig(**values))


def rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean(dim=-1).sqrt()


def test_has_particle_adapter_compatible_interface_and_no_trainable_state() -> None:
    intervention = make_intervention()
    hidden = torch.randn(2, 4, 16)
    output = intervention(
        hidden,
        query_state=torch.randn(2, 16),
        particle_z=torch.randn(2, 16),
        response_mask=torch.ones(2, 4, dtype=torch.bool),
    )

    assert isinstance(output, ParticleAdapterOutput)
    assert intervention.hidden_size == 16
    assert intervention.latent_size == 16
    assert list(intervention.parameters()) == []
    assert set(intervention.state_dict()) == {"relative_rms_scale"}
    assert intervention.relative_rms_scale.dtype == torch.float32


def test_scale_can_be_swept_and_zero_scale_is_exact_identity() -> None:
    intervention = make_intervention(relative_rms_scale=0.03)
    hidden = torch.randn(2, 3, 16)
    intervention.set_relative_rms_scale(0.0)
    output = intervention(hidden, torch.randn(2, 16), torch.randn(2, 16))

    assert torch.equal(output.hidden_states, hidden)
    assert torch.count_nonzero(output.delta) == 0
    assert float(intervention.relative_rms_scale) == 0.0


def test_zero_particle_is_bit_exact_identity_in_mixed_batch() -> None:
    intervention = make_intervention()
    hidden = torch.randn(3, 5, 16)
    particles = torch.randn(3, 16)
    particles[0].zero_()

    output = intervention(
        hidden,
        torch.randn(3, 16),
        particles,
        torch.ones(3, 5, dtype=torch.bool),
    )

    assert torch.equal(output.hidden_states[0], hidden[0])
    assert torch.count_nonzero(output.delta[0]) == 0
    assert torch.count_nonzero(output.relative_rms[0]) == 0
    assert torch.count_nonzero(output.delta[1:]) > 0


def test_false_mask_positions_are_bit_exact_identity() -> None:
    intervention = make_intervention()
    hidden = torch.randn(2, 6, 16)
    response_mask = torch.tensor(
        [[False, False, False, True, True, True], [False, False, True, True, True, True]]
    )

    output = intervention(hidden, torch.randn(2, 16), torch.randn(2, 16), response_mask)

    assert torch.equal(output.hidden_states[~response_mask], hidden[~response_mask])
    assert torch.count_nonzero(output.delta[~response_mask]) == 0
    assert torch.count_nonzero(output.relative_rms[~response_mask]) == 0


def test_all_false_mask_disables_intervention_exactly() -> None:
    intervention = make_intervention()
    hidden = torch.randn(2, 4, 16)

    output = intervention(
        hidden,
        torch.randn(2, 16),
        torch.randn(2, 16),
        torch.zeros(2, 4, dtype=torch.bool),
    )

    assert torch.equal(output.hidden_states, hidden)
    assert torch.count_nonzero(output.delta) == 0
    assert torch.count_nonzero(output.relative_rms) == 0


@pytest.mark.parametrize("scale", [0.005, 0.01, 0.03, 0.05])
def test_active_delta_has_configured_relative_rms(scale: float) -> None:
    intervention = make_intervention(relative_rms_scale=scale)
    hidden = torch.randn(7, 9, 16) + 0.25

    output = intervention(
        hidden,
        torch.randn(7, 16),
        torch.randn(7, 16),
        torch.ones(7, 9, dtype=torch.bool),
    )
    measured = rms(output.delta) / rms(hidden)

    assert torch.allclose(measured, torch.full_like(measured, scale), atol=2e-6, rtol=2e-5)
    assert output.amplitude.shape == (7, 1, 1)
    assert torch.equal(output.amplitude, torch.full_like(output.amplitude, scale))


def test_particle_direction_is_hidden_sized_normalized_and_query_independent() -> None:
    intervention = make_intervention(relative_rms_scale=0.04)
    hidden = torch.ones(2, 3, 16)
    particle = torch.linspace(-2.0, 1.0, 16).repeat(2, 1)
    queries_a = torch.stack([torch.ones(16), -torch.ones(16)])
    queries_b = torch.randn(2, 16)

    output_a = intervention(hidden, queries_a, particle)
    output_b = intervention(hidden, queries_b, particle)

    assert torch.equal(output_a.delta, output_b.delta)
    # One particle direction is reused across all sequence positions.
    assert torch.equal(output_a.delta[:, 0], output_a.delta[:, 1])
    assert torch.allclose(rms(output_a.delta), torch.full((2, 3), 0.04), atol=2e-6)


def test_bf16_path_preserves_anchor_and_approximately_preserves_scale() -> None:
    intervention = make_intervention(relative_rms_scale=0.03)
    hidden = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    particles = torch.randn(2, 16, dtype=torch.bfloat16)
    particles[0].zero_()

    output = intervention(hidden, torch.randn(2, 16), particles)

    assert output.delta.dtype == torch.bfloat16
    assert output.hidden_states.dtype == torch.bfloat16
    assert torch.equal(output.hidden_states[0], hidden[0])
    measured = rms(output.delta[1]) / rms(hidden[1])
    assert torch.allclose(measured, torch.full_like(measured, 0.03), atol=2e-4, rtol=1e-2)


def test_query_receives_no_gradient_and_reference_rms_is_detached_by_default() -> None:
    intervention = make_intervention()
    hidden = torch.randn(2, 4, 16, requires_grad=True)
    query = torch.randn(2, 16, requires_grad=True)

    output = intervention(hidden, query, torch.randn(2, 16))
    output.hidden_states.sum().backward()

    assert query.grad is None
    # With a detached scale, the only hidden-state gradient is the identity path.
    assert torch.equal(hidden.grad, torch.ones_like(hidden))


def test_single_particle_broadcasts_across_batch() -> None:
    intervention = make_intervention()
    hidden = torch.ones(3, 2, 16)
    particle = torch.randn(16)

    output = intervention(hidden, torch.randn(3, 16), particle)

    assert torch.equal(output.delta[0], output.delta[1])
    assert torch.equal(output.delta[1], output.delta[2])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"relative_rms_scale": -0.01}, "relative_rms_scale"),
        ({"relative_rms_scale": 1.01}, "relative_rms_scale"),
        ({"rms_eps": 0.0}, "rms_eps"),
    ],
)
def test_invalid_config_is_rejected(changes: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_intervention(**changes)


def test_particle_must_be_hidden_sized() -> None:
    intervention = make_intervention()

    with pytest.raises(ValueError, match=r"particle_z.*\[batch, 16\]"):
        intervention(torch.randn(2, 3, 16), torch.randn(2, 16), torch.randn(2, 4))


def test_query_shape_is_still_validated_for_drop_in_compatibility() -> None:
    intervention = make_intervention()

    with pytest.raises(ValueError, match="query_state"):
        intervention(torch.randn(2, 3, 16), torch.randn(2, 4), torch.randn(2, 16))
