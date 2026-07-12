from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from hrm_particle.adapter import ParticleAdapter, ParticleAdapterConfig, SharedQHead


def make_adapter(**overrides) -> ParticleAdapter:
    values = dict(
        hidden_size=16,
        latent_size=4,
        bottleneck_size=8,
        max_relative_rms=0.10,
        initial_relative_rms=0.03,
    )
    values.update(overrides)
    return ParticleAdapter(ParticleAdapterConfig(**values))


def rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean(dim=-1).sqrt()


def test_zero_particle_is_bit_exact_identity() -> None:
    adapter = make_adapter()
    hidden = torch.randn(3, 5, 16)
    query = torch.randn(3, 16)
    zero_z = torch.zeros(3, 4)

    output = adapter(hidden, query, zero_z, torch.ones(3, 5, dtype=torch.bool))

    assert torch.count_nonzero(output.delta) == 0
    assert torch.equal(output.hidden_states, hidden)
    assert torch.count_nonzero(output.relative_rms) == 0


def test_relative_rms_backward_is_finite_with_zero_anchor_and_masked_prompt() -> None:
    """Regression: sqrt(RMS) at exact zero previously produced 0*inf NaNs."""

    adapter = make_adapter()
    hidden = torch.randn(2, 5, 16)
    query = torch.randn(2, 16)
    latents = torch.randn(2, 4)
    latents[0].zero_()
    response_mask = torch.tensor(
        [[False, False, True, True, True], [False, False, True, True, True]]
    )

    output = adapter(hidden, query, latents, response_mask)
    assert torch.count_nonzero(output.relative_rms[0]) == 0
    loss = output.relative_rms[1, response_mask[1]].square().mean()
    loss.backward()

    gradients = [parameter.grad for parameter in adapter.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_masked_prompt_positions_are_bit_exact() -> None:
    adapter = make_adapter()
    hidden = torch.randn(2, 6, 16)
    query = torch.randn(2, 16)
    z = torch.randn(2, 4)
    response_mask = torch.tensor(
        [[False, False, False, True, True, True], [False, False, True, True, True, True]]
    )

    output = adapter(hidden, query, z, response_mask)

    assert torch.equal(output.hidden_states[~response_mask], hidden[~response_mask])
    assert torch.count_nonzero(output.delta[~response_mask]) == 0
    assert torch.count_nonzero(output.delta[response_mask]) > 0


def test_injection_is_bounded_by_relative_rms() -> None:
    maximum = 0.075
    adapter = make_adapter(max_relative_rms=maximum, initial_relative_rms=0.02)
    hidden = torch.randn(7, 9, 16) + 0.5
    query = torch.randn(7, 16)
    z = torch.randn(7, 4)

    output = adapter(hidden, query, z, torch.ones(7, 9, dtype=torch.bool))
    measured = rms(output.delta) / rms(hidden)

    assert torch.all(measured <= maximum + 1e-6)
    assert output.amplitude.max().item() <= maximum
    assert output.amplitude.min().item() > 0


def test_direction_depends_on_query_and_particle_code() -> None:
    adapter = make_adapter()
    hidden = torch.ones(2, 3, 16)
    z = torch.tensor([[1.0, -0.5, 0.2, 0.7], [1.0, -0.5, 0.2, 0.7]])
    query = torch.stack([torch.ones(16), -torch.ones(16)])

    output = adapter(hidden, query, z, torch.ones(2, 3, dtype=torch.bool))

    assert not torch.allclose(output.delta[0], output.delta[1])


def test_query_is_stop_gradient_but_adapter_receives_gradients() -> None:
    adapter = make_adapter()
    hidden = torch.randn(2, 4, 16, requires_grad=True)
    query = torch.randn(2, 16, requires_grad=True)
    z = torch.randn(2, 4)

    output = adapter(hidden, query, z, torch.ones(2, 4, dtype=torch.bool))
    output.hidden_states.square().mean().backward()

    assert query.grad is None
    assert hidden.grad is not None
    assert adapter.output_proj.weight.grad is not None
    assert adapter.output_proj.weight.grad.abs().sum() > 0


def test_full_history_mask_aligns_to_cached_suffix() -> None:
    adapter = make_adapter()
    hidden = torch.randn(2, 1, 16)
    query = torch.randn(2, 16)
    z = torch.randn(2, 4)
    full_history_mask = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1]])

    output = adapter(hidden, query, z, full_history_mask)

    assert torch.count_nonzero(output.delta) > 0


def test_shared_q_head_is_query_aware_and_trainable() -> None:
    head = SharedQHead(hidden_size=16, bottleneck_size=8)
    terminal = torch.randn(2, 16)
    # Identical terminal states isolate the query contribution.
    terminal[1] = terminal[0]
    query = torch.stack([torch.ones(16), -torch.ones(16)])

    scores = head(terminal, query)

    assert scores.shape == (2,)
    assert not torch.allclose(scores[0], scores[1])
    loss = F.binary_cross_entropy_with_logits(scores, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert head.query_proj.weight.grad is not None
    assert head.output_proj.weight.grad is not None


def test_shared_q_head_constant_prior_ties_and_argmax_falls_back_to_branch_zero() -> None:
    head = SharedQHead(hidden_size=16, bottleneck_size=8)
    head.initialize_constant_prior(0.2)

    scores = head(torch.randn(4, 16), torch.randn(4, 16))

    expected = torch.full_like(scores, torch.logit(torch.tensor(0.2)))
    assert torch.allclose(scores, expected)
    assert torch.count_nonzero(head.output_proj.weight) == 0
    assert scores.argmax().item() == 0


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_shared_q_head_constant_prior_rejects_invalid_probability(probability: float) -> None:
    head = SharedQHead(hidden_size=16, bottleneck_size=8)
    with pytest.raises(ValueError, match="strictly between"):
        head.initialize_constant_prior(probability)


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": 0},
        {"latent_size": 0},
        {"max_relative_rms": 0.0},
        {"initial_relative_rms": 0.10},
    ],
)
def test_invalid_config_is_rejected(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        make_adapter(**changes)
