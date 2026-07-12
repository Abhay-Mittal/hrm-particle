from __future__ import annotations

import pytest
import torch

from hrm_particle.objectives import (
    anchor_rescue_advantages,
    clipped_token_policy_loss,
    supervised_q_loss,
)


def test_anchor_rescue_credit_excludes_anchor_and_rewards_unique_rescue():
    rewards = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    advantages = anchor_rescue_advantages(rewards, alpha=0.2)
    assert torch.equal(advantages[:, 0], torch.zeros(3))
    assert advantages[0].tolist() == pytest.approx([0.0, 1.4, -0.4, -0.4])
    assert advantages[1].tolist() == pytest.approx([0.0, 0.8, -0.4, -0.4])
    assert torch.equal(advantages[2], torch.zeros(4))


def test_token_level_ppo_clips_both_advantage_signs_and_length_normalizes():
    # Sequence 0 has positive advantage and ratio 1.5 -> clipped to 1.2.
    # Sequence 1 has negative advantage and ratio 0.5 -> clipped to 0.8.
    ratios = torch.tensor([[[1.5, 1.5, 99.0], [0.5, 99.0, 99.0]]])
    new = ratios.log()
    old = torch.zeros_like(new)
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.tensor([[[True, True, False], [True, False, False]]])
    output = clipped_token_policy_loss(
        new,
        old,
        advantages,
        mask,
        clip_epsilon=0.2,
    )
    # Per-sequence means are -1.2 and +0.8, then the two sequences are averaged.
    assert float(output.policy_loss) == pytest.approx(-0.2, abs=1e-6)
    assert float(output.clip_fraction) == pytest.approx(1.0)
    assert int(output.valid_sequences) == 2


def test_masked_first_token_has_zero_actor_gradient():
    new = torch.zeros(1, 2, 2, requires_grad=True)
    old = torch.zeros_like(new)
    advantages = torch.tensor([[0.0, 1.0]])
    # Token one is generated from shared clean prefill and must be excluded.
    mask = torch.tensor([[[False, False], [False, True]]])
    loss = clipped_token_policy_loss(new, old, advantages, mask).loss
    loss.backward()
    assert new.grad is not None
    assert float(new.grad[0, 1, 0]) == 0.0
    assert float(new.grad[0, 1, 1]) != 0.0


def test_reference_k3_is_nonnegative_and_zero_only_at_equal_logprob():
    behavior = torch.zeros(1, 1, 3)
    new = torch.zeros(1, 1, 3, requires_grad=True)
    raw = torch.tensor([[[0.0, 0.4, -0.4]]], requires_grad=True)
    reference = torch.zeros_like(raw)
    mask = torch.ones_like(new, dtype=torch.bool)
    result = clipped_token_policy_loss(
        new,
        behavior,
        torch.zeros(1, 1),
        mask,
        reference_logprobs=reference,
        kl_logprobs=raw,
        kl_coefficient=1.0,
    )
    expected = (torch.exp(-raw.detach()) - 1.0 + raw.detach()).mean()
    assert float(result.approx_kl) == pytest.approx(float(expected))
    assert float(result.approx_kl) >= 0.0
    result.loss.backward()
    # Gradient sign moves positive and negative log-ratios toward zero.
    assert raw.grad is not None
    assert float(raw.grad[0, 0, 1]) > 0.0
    assert float(raw.grad[0, 0, 2]) < 0.0


def test_q_loss_uses_mixed_groups_for_ranking_and_uniform_groups_for_bce():
    logits = torch.tensor([[2.0, -1.0, -2.0], [0.2, 0.1, 0.3]], requires_grad=True)
    rewards = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    output = supervised_q_loss(logits, rewards, ranking_weight=0.1)
    assert output.ranking_pairs == 2
    assert float(output.bce_loss) > 0.0
    output.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_objective_shape_validation():
    with pytest.raises(ValueError):
        anchor_rescue_advantages(torch.zeros(4))
    with pytest.raises(ValueError):
        clipped_token_policy_loss(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2, 3, dtype=torch.bool),
        )
