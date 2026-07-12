"""Losses and credit assignment for adapter-only verifier RL.

All actor rewards entering this module are external verifier labels.  The Q
head has a separate supervised objective and is intentionally absent from the
policy objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class PolicyLossOutput:
    loss: Tensor
    policy_loss: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor
    mean_ratio: Tensor
    valid_sequences: Tensor


@dataclass(frozen=True)
class QLossOutput:
    loss: Tensor
    bce_loss: Tensor
    ranking_loss: Tensor
    ranking_pairs: int


def _require_shape(name: str, value: Tensor, expected_ndim: int) -> None:
    if value.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions, got {tuple(value.shape)}")


def anchor_rescue_advantages(rewards: Tensor, alpha: float = 0.2) -> Tensor:
    """Return leave-one-out advantages for a fixed anchor plus explorers.

    ``rewards`` has shape ``[batch, K]`` and particle zero is the untrained
    anchor.  Its advantage is exactly zero.  Explorer credit combines ordinary
    leave-one-out correctness with a bonus for being the unique explorer that
    rescues a prompt on which the anchor failed::

        A_i = (1-alpha) (r_i - mean(r_-i))
              + alpha * (K-1) * I(anchor wrong, i uniquely correct)

    Multiplication by ``K-1`` makes the rescue term retain weight after a loss
    that averages over explorers.
    """

    _require_shape("rewards", rewards, 2)
    if rewards.shape[1] < 2:
        raise ValueError("K must be at least 2 (one anchor and one explorer)")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    rewards = rewards.float()
    explorers = rewards[:, 1:]
    n_explorers = explorers.shape[1]
    explorer_sum = explorers.sum(dim=1, keepdim=True)
    if n_explorers == 1:
        # There is no peer baseline for K=2.
        mean_credit = explorers
    else:
        peer_mean = (explorer_sum - explorers) / float(n_explorers - 1)
        mean_credit = explorers - peer_mean

    anchor_wrong = (rewards[:, :1] < 0.5).to(explorers.dtype)
    unique_rescue = anchor_wrong * (
        (explorers > 0.5) & ((explorer_sum - explorers) < 0.5)
    ).to(explorers.dtype)
    explorer_advantage = (1.0 - alpha) * mean_credit + (
        alpha * float(n_explorers) * unique_rescue
    )

    return torch.cat((torch.zeros_like(rewards[:, :1]), explorer_advantage), dim=1)


def set_coverage_advantages(rewards: Tensor, alpha: float = 0.2) -> Tensor:
    """General all-particle leave-one-out set-coverage credit.

    This helper is useful for ablations.  The POC trainer uses
    :func:`anchor_rescue_advantages` because particle zero is fixed.
    """

    _require_shape("rewards", rewards, 2)
    if rewards.shape[1] < 2:
        raise ValueError("K must be at least 2")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    rewards = rewards.float()
    k = rewards.shape[1]
    total = rewards.sum(dim=1, keepdim=True)
    mean_advantage = rewards - (total - rewards) / float(k - 1)
    unique_success = ((rewards > 0.5) & ((total - rewards) < 0.5)).to(rewards.dtype)
    return (1.0 - alpha) * mean_advantage + alpha * float(k) * unique_success


def masked_sequence_mean(values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Mean over tokens for each sequence, plus its validity mask.

    This is the explicit length normalization used by the actor: a 100-token
    response and a 10-token response receive equal sequence weight.
    """

    if values.shape != mask.shape:
        raise ValueError(f"values and mask must match, got {values.shape} and {mask.shape}")
    weights = mask.to(values.dtype)
    lengths = weights.sum(dim=-1)
    valid = lengths > 0
    means = (values * weights).sum(dim=-1) / lengths.clamp_min(1.0)
    return means, valid


def clipped_token_policy_loss(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_epsilon: float = 0.2,
    reference_logprobs: Optional[Tensor] = None,
    kl_logprobs: Optional[Tensor] = None,
    kl_coefficient: float = 0.0,
) -> PolicyLossOutput:
    """Token-ratio PPO with per-response length normalization.

    Inputs have shapes ``[batch, K, time]`` except ``advantages`` which is
    ``[batch, K]``.  Ratios are computed *per token*; multiplying ratios across
    a complete response would be numerically unstable and is intentionally not
    supported.  Particle zero is excluded by forcing its action mask to false in
    the trainer (the objective itself remains general).
    """

    _require_shape("new_logprobs", new_logprobs, 3)
    if old_logprobs.shape != new_logprobs.shape or action_mask.shape != new_logprobs.shape:
        raise ValueError("new_logprobs, old_logprobs, and action_mask must have identical shapes")
    if advantages.shape != new_logprobs.shape[:2]:
        raise ValueError("advantages must have shape [batch, K]")
    if clip_epsilon <= 0.0:
        raise ValueError("clip_epsilon must be positive")
    if kl_coefficient < 0.0:
        raise ValueError("kl_coefficient must be non-negative")

    log_ratio = new_logprobs - old_logprobs.detach()
    # Avoid inf from pathological logits without changing the useful PPO range.
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    token_advantages = advantages.unsqueeze(-1).to(new_logprobs.dtype)
    surrogate = torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)
    token_policy_loss = -surrogate

    sequence_policy_loss, valid = masked_sequence_mean(token_policy_loss, action_mask)
    if not bool(valid.any()):
        # Keep a differentiable zero so callers can safely invoke backward.
        policy_loss = new_logprobs.sum() * 0.0
    else:
        policy_loss = sequence_policy_loss[valid].mean()

    # Schulman's non-negative sampled KL approximation relative to the rollout
    # policy is used for diagnostics.
    token_approx_kl = (ratio - 1.0) - log_ratio
    sequence_approx_kl, _ = masked_sequence_mean(token_approx_kl, action_mask)
    approx_kl = sequence_approx_kl[valid].mean() if bool(valid.any()) else policy_loss.detach()

    if reference_logprobs is not None:
        if reference_logprobs.shape != new_logprobs.shape:
            raise ValueError("reference_logprobs must match new_logprobs")
        if kl_logprobs is not None and kl_logprobs.shape != new_logprobs.shape:
            raise ValueError("kl_logprobs must match new_logprobs")
        # PPO ratios use the exact temperature/top-p behavior policy, while KL
        # uses raw model log-probabilities. A top-p reference can assign -inf to
        # an exploratory action and destabilize training.
        actor_kl_logprobs = new_logprobs if kl_logprobs is None else kl_logprobs
        log_ratio_reference = actor_kl_logprobs - reference_logprobs.detach()
        # Non-negative k3 estimator.  Plain ``new-ref`` is noisy, can be
        # negative on a finite batch, and makes the reported penalty misleading.
        token_reference_kl = (
            torch.exp((-log_ratio_reference).clamp(min=-20.0, max=20.0))
            - 1.0
            + log_ratio_reference
        )
        sequence_reference_kl, _ = masked_sequence_mean(token_reference_kl, action_mask)
        reference_kl = (
            sequence_reference_kl[valid].mean() if bool(valid.any()) else policy_loss.detach()
        )
    else:
        reference_kl = approx_kl

    total_loss = policy_loss + kl_coefficient * reference_kl
    clipped = ((ratio - 1.0).abs() > clip_epsilon).to(new_logprobs.dtype)
    sequence_clip, _ = masked_sequence_mean(clipped, action_mask)
    clip_fraction = sequence_clip[valid].mean() if bool(valid.any()) else policy_loss.detach()
    sequence_ratio, _ = masked_sequence_mean(ratio, action_mask)
    mean_ratio = sequence_ratio[valid].mean() if bool(valid.any()) else ratio.new_tensor(1.0)

    return PolicyLossOutput(
        loss=total_loss,
        policy_loss=policy_loss,
        approx_kl=reference_kl,
        clip_fraction=clip_fraction,
        mean_ratio=mean_ratio,
        valid_sequences=valid.sum(),
    )


def supervised_q_loss(
    logits: Tensor,
    rewards: Tensor,
    *,
    ranking_weight: float = 0.1,
) -> QLossOutput:
    """BCE plus within-prompt correct-vs-wrong pairwise ranking loss."""

    if logits.shape != rewards.shape or logits.ndim != 2:
        raise ValueError("logits and rewards must both have shape [batch, K]")
    if ranking_weight < 0.0:
        raise ValueError("ranking_weight must be non-negative")

    labels = rewards.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    ranking_terms = []
    ranking_pairs = 0
    for group_logits, group_labels in zip(logits, labels):
        positive = group_logits[group_labels > 0.5]
        negative = group_logits[group_labels <= 0.5]
        if positive.numel() and negative.numel():
            differences = positive[:, None] - negative[None, :]
            ranking_terms.append(F.softplus(-differences).mean())
            ranking_pairs += int(differences.numel())

    if ranking_terms:
        ranking = torch.stack(ranking_terms).mean()
    else:
        ranking = logits.sum() * 0.0
    return QLossOutput(
        loss=bce + ranking_weight * ranking,
        bce_loss=bce,
        ranking_loss=ranking,
        ranking_pairs=ranking_pairs,
    )


__all__ = [
    "PolicyLossOutput",
    "QLossOutput",
    "anchor_rescue_advantages",
    "clipped_token_policy_loss",
    "masked_sequence_mean",
    "set_coverage_advantages",
    "supervised_q_loss",
]
