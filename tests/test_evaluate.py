from __future__ import annotations

import pytest
import torch
from torch import nn

from hrm_particle.evaluate import evaluate_examples, evaluate_rollout, paired_bootstrap_delta
from hrm_particle.rollout import ParticleRollout, RolloutExample
from hrm_particle.verifier import ExactArithmeticVerifier


def _rollout() -> ParticleRollout:
    rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    b, k = rewards.shape
    sequence, time, hidden = 4, 2, 3
    verifier = ExactArithmeticVerifier()
    texts = [["1", "2", "1", "1"], ["3", "3", "3", "3"]]
    references = ["2", "3"]
    verification = [
        [verifier.verify(text, answer) for text in group]
        for group, answer in zip(texts, references)
    ]
    return ParticleRollout(
        model_input_ids=torch.zeros(b, k, sequence, dtype=torch.long),
        attention_mask=torch.ones(b, k, sequence, dtype=torch.bool),
        token_type_ids=torch.zeros(b, k, sequence, dtype=torch.long),
        position_ids=torch.arange(sequence).view(1, 1, -1).expand(b, k, -1),
        particle_mask=torch.ones(b, k, sequence, dtype=torch.bool),
        particle_z=torch.zeros(b, k, 2),
        action_ids=torch.zeros(b, k, time, dtype=torch.long),
        generated_mask=torch.ones(b, k, time, dtype=torch.bool),
        action_mask=torch.ones(b, k, time, dtype=torch.bool),
        action_positions=torch.zeros(b, k, time, dtype=torch.long),
        old_logprobs=torch.zeros(b, k, time),
        rewards=rewards,
        prompt_summary=torch.zeros(b, k, hidden),
        terminal_states=torch.zeros(b, k, hidden),
        response_texts=texts,
        example_ids=["a", "b"],
        references=references,
        verification=verification,
        temperature=0.8,
        top_p=0.95,
    )


def test_generation_and_rescue_metrics_are_separate_from_q():
    result = evaluate_rollout(_rollout())
    assert result.metrics["anchor_accuracy"] == pytest.approx(0.5)
    assert result.metrics["explorer_accuracy"] == pytest.approx(4 / 6)
    assert result.metrics["oracle_pass_at_k"] == 1.0
    assert result.metrics["rescue_given_anchor_wrong"] == 1.0
    assert result.metrics["parseable_fraction"] == 1.0
    assert result.metrics["duplicate_candidate_fraction"] == pytest.approx(5 / 6)
    assert result.metrics["all_candidates_identical_fraction"] == 0.5
    assert "q_selected_accuracy" not in result.metrics


def test_q_metrics_include_capture_regret_pairwise_and_calibration():
    # Select branch 1 in the mixed first prompt and any branch in all-correct second.
    q_logits = torch.tensor([[0.0, 5.0, -2.0, -3.0], [0.0, 0.0, 0.0, 0.0]])
    result = evaluate_rollout(_rollout(), q_logits)
    assert result.metrics["q_selected_accuracy"] == 1.0
    assert result.metrics["q_oracle_regret"] == 0.0
    assert result.metrics["q_selected_mixed_accuracy"] == 1.0
    assert result.metrics["q_within_prompt_pair_accuracy"] == 1.0
    assert 0.0 <= result.metrics["q_brier"] <= 1.0
    assert 0.0 <= result.metrics["q_ece_10bin"] <= 1.0


def test_paired_bootstrap_uses_prompt_pairing_and_is_reproducible():
    system = [1, 1, 0, 1, 1]
    baseline = [0, 1, 0, 0, 1]
    first = paired_bootstrap_delta(system, baseline, num_samples=500, seed=7)
    second = paired_bootstrap_delta(system, baseline, num_samples=500, seed=7)
    assert first == second
    assert first.mean_delta == pytest.approx(0.4)
    assert first.low <= first.mean_delta <= first.high


def _single_group(reward_values, q_values, example_id):
    rewards = torch.tensor([reward_values], dtype=torch.float32)
    b, k = rewards.shape
    verifier = ExactArithmeticVerifier()
    texts = [["2" if reward else "1" for reward in reward_values]]
    terminal = torch.zeros(b, k, 3)
    terminal[0, :, 0] = torch.tensor(q_values)
    return ParticleRollout(
        model_input_ids=torch.zeros(b, k, 3, dtype=torch.long),
        attention_mask=torch.ones(b, k, 3, dtype=torch.bool),
        token_type_ids=torch.zeros(b, k, 3, dtype=torch.long),
        position_ids=torch.arange(3).view(1, 1, -1).expand(b, k, -1),
        particle_mask=torch.ones(b, k, 3, dtype=torch.bool),
        particle_z=torch.zeros(b, k, 2),
        action_ids=torch.zeros(b, k, 1, dtype=torch.long),
        generated_mask=torch.ones(b, k, 1, dtype=torch.bool),
        action_mask=torch.ones(b, k, 1, dtype=torch.bool),
        action_positions=torch.zeros(b, k, 1, dtype=torch.long),
        old_logprobs=torch.zeros(b, k, 1),
        rewards=rewards,
        prompt_summary=torch.zeros(b, k, 3),
        terminal_states=terminal,
        response_texts=texts,
        example_ids=[example_id],
        references=["2"],
        verification=[[verifier.verify(text, "2") for text in texts[0]]],
        temperature=1.0,
        top_p=1.0,
    )


def test_global_pair_accuracy_excludes_uniform_groups_and_weights_actual_pairs():
    rollouts = [
        _single_group([1, 0, 0, 0], [4, 3, 2, 1], "one-positive"),  # 3/3
        _single_group([1, 1, 0, 0], [0, 3, 2, 1], "two-positive"),  # 2/4
        _single_group([1, 1, 1, 1], [0, 0, 0, 0], "uniform"),  # excluded
    ]

    class Engine:
        def __init__(self):
            self.index = 0

        def generate(self, _examples):
            rollout = rollouts[self.index]
            self.index += 1
            return rollout

    class Q(nn.Module):
        def forward(self, terminal, prompt):
            return terminal[:, 0]

    result = evaluate_examples(
        Engine(),
        [RolloutExample("q", "2", str(index)) for index in range(3)],
        q_head=Q(),
        batch_size=1,
    )
    assert result.metrics["q_pair_count"] == 7.0
    assert result.metrics["q_within_prompt_pair_accuracy"] == pytest.approx(5 / 7)
