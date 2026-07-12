from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hrm_particle.rollout import (
    ParticleRolloutEngine,
    RolloutConfig,
    RolloutExample,
    score_actions,
)
from hrm_particle.trainer import ParticleTrainer, TrainerConfig
from hrm_particle.trainer import train_from_config


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 4 for char in text]

    def decode(self, ids, skip_special_tokens=True):
        characters = []
        for token in ids:
            if token in {self.pad_token_id, self.eos_token_id} and skip_special_tokens:
                continue
            if token >= 4:
                characters.append(chr(token - 4))
        return "".join(characters)


class DummyParticlePolicy(nn.Module):
    """Tiny differentiable policy implementing the rollout wrapper contract."""

    def __init__(self, latent_dim=2, hidden_size=4, vocab_size=260):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.particle_adapter = nn.Linear(latent_dim, 1, bias=False)
        nn.init.zeros_(self.particle_adapter.weight)
        with torch.no_grad():
            self.particle_adapter.weight[0, 0] = 2.0
        self.frozen_embedding = nn.Embedding(vocab_size, hidden_size)
        self.frozen_embedding.weight.requires_grad_(False)
        self.prefix_id = ord("§") + 4
        self.one_id = ord("1") + 4
        self.two_id = ord("2") + 4

    def forward(
        self,
        input_ids,
        attention_mask,
        particle_z=None,
        particle_mask=None,
        prompt_summary=None,
        query_state=None,
        **kwargs,
    ):
        hidden = self.frozen_embedding(input_ids)
        batch, sequence = input_ids.shape
        query = query_state if query_state is not None else prompt_summary
        if query is None:
            query = hidden[:, -1].detach()
        logits = torch.full(
            (batch, sequence, self.vocab_size),
            -12.0,
            device=input_ids.device,
            dtype=hidden.dtype,
        )
        # Default and post-answer behavior terminates.
        logits[..., 1] = 8.0
        delta = torch.zeros_like(hidden)
        relative_rms = torch.zeros(batch, sequence, 1, device=input_ids.device)
        if particle_z is not None:
            steering = self.particle_adapter(particle_z).squeeze(-1)
            aligned_mask = particle_mask[:, -sequence:].bool()
            delta = aligned_mask[..., None] * steering[:, None, None]
            hidden = hidden + delta
            relative_rms = aligned_mask[..., None] * steering[:, None, None].abs() / 10.0
        else:
            steering = torch.zeros(batch, device=input_ids.device)

        response_start = input_ids.eq(self.prefix_id)
        if bool(response_start.any()):
            rows, columns = response_start.nonzero(as_tuple=True)
            local_steering = steering[rows]
            logits[rows, columns, 1] = -12.0
            logits[rows, columns, self.one_id] = 1.0 - local_steering
            logits[rows, columns, self.two_id] = local_steering

        lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        terminal = hidden[torch.arange(batch), lengths]
        return SimpleNamespace(
            logits=logits,
            past_key_values=None,
            terminal_state=terminal,
            prompt_summary=query,
            injection_delta=delta if particle_z is not None else None,
            relative_rms=relative_rms if particle_z is not None else None,
        )


class DummyQHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, terminal_state, prompt_summary):
        return self.linear(torch.cat((terminal_state, prompt_summary), dim=-1)).squeeze(-1)


def fixed_latents(batch_size, k, latent_dim, **kwargs):
    result = torch.zeros(batch_size, k, latent_dim, device=kwargs["device"], dtype=kwargs["dtype"])
    result[:, 1, 0] = 1.0
    result[:, 2:, 0] = -1.0
    return result


def _engine(model, *, seed=5, latent_sampler=fixed_latents):
    return ParticleRolloutEngine(
        model,
        CharTokenizer(),
        config=RolloutConfig(
            k=4,
            latent_dim=2,
            max_new_tokens=2,
            temperature=1.0,
            top_p=1.0,
            response_prefix="§",
            first_token_mode="causal_prefix",
            use_cache=False,
            compute_reference_logprobs=True,
            seed=seed,
        ),
        latent_sampler=latent_sampler,
    )


@pytest.mark.integration
def test_two_prompt_k4_rollout_keeps_branch_rewards_and_replay_aligned() -> None:
    model = DummyParticlePolicy()
    engine = _engine(model)
    examples = [
        RolloutExample("First prompt", "2", "first"),
        RolloutExample("Second prompt", "1", "second"),
    ]
    rollout = engine.generate(examples)

    assert rollout.rewards.shape == (2, 4)
    assert rollout.particle_z.shape == (2, 4, 2)
    assert rollout.example_ids == ["first", "second"]
    for batch_index, example in enumerate(examples):
        for branch in range(4):
            expected = float(rollout.response_texts[batch_index][branch] == example.answer)
            assert float(rollout.rewards[batch_index, branch]) == expected
            assert rollout.verification[batch_index][branch].correct == bool(expected)

    replay = score_actions(model, rollout)
    valid = rollout.generated_mask
    ratios = torch.exp(replay.logprobs[valid] - rollout.old_logprobs[valid])
    torch.testing.assert_close(ratios, torch.ones_like(ratios), atol=1e-6, rtol=1e-6)


@pytest.mark.integration
def test_full_cpu_rollout_replay_actor_and_detached_q_update():
    model = DummyParticlePolicy()
    q_head = DummyQHead(model.hidden_size)
    engine = _engine(model)
    actor_optimizer = torch.optim.Adam(model.particle_adapter.parameters(), lr=0.05)
    q_optimizer = torch.optim.Adam(q_head.parameters(), lr=0.01)
    trainer = ParticleTrainer(
        model,
        engine,
        actor_optimizer,
        q_optimizer,
        q_head=q_head,
        config=TrainerConfig(kl_coefficient=0.0, injection_penalty_coefficient=0.0),
    )
    before_actor = model.particle_adapter.weight.detach().clone()
    before_q = q_head.linear.weight.detach().clone()
    result = trainer.train_batch([RolloutExample("What is one plus one?", "2", "dummy")])

    rollout = result.rollout
    assert rollout.rewards.tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert torch.equal(rollout.particle_z[:, 0], torch.zeros_like(rollout.particle_z[:, 0]))
    assert rollout.action_mask[0, 1, 0]  # causal prefix branches the first sampled token
    assert not rollout.particle_mask[0, 1][rollout.token_type_ids[0, 1] == 1].any()
    assert not torch.equal(before_actor, model.particle_adapter.weight)
    assert not torch.equal(before_q, q_head.linear.weight)
    assert result.metrics["reward/oracle_pass_at_k"] == 1.0
    assert result.metrics["q/ranking_pairs"] == 3.0
    assert torch.isfinite(torch.tensor(list(result.metrics.values()))).all()


@pytest.mark.integration
def test_actor_rewards_and_q_correctness_labels_are_independent_targets():
    model = DummyParticlePolicy()
    q_head = DummyQHead(model.hidden_size)
    nn.init.zeros_(q_head.linear.weight)
    nn.init.zeros_(q_head.linear.bias)
    engine = _engine(model)
    actor_optimizer = torch.optim.SGD(model.particle_adapter.parameters(), lr=0.05)
    q_optimizer = torch.optim.SGD(q_head.parameters(), lr=0.1)
    trainer = ParticleTrainer(
        model,
        engine,
        actor_optimizer,
        q_optimizer,
        q_head=q_head,
        config=TrainerConfig(kl_coefficient=0.0, injection_penalty_coefficient=0.0),
    )
    rollout = engine.generate([RolloutExample("What is one plus one?", "2", "dummy")])
    assert rollout.rewards.tolist() == [[0.0, 1.0, 0.0, 0.0]]
    rollout = replace(rollout, q_labels=torch.ones_like(rollout.rewards))
    before_actor = model.particle_adapter.weight.detach().clone()
    before_q_bias = q_head.linear.bias.detach().clone()

    metrics = trainer.train_rollout(rollout)

    # All-one Q labels give the zero-initialized Q bias a negative gradient,
    # hence an SGD update increases it.  They would give the actor no relative
    # credit, so its update proves actor advantages still came from rewards.
    assert q_head.linear.bias.item() > before_q_bias.item()
    assert not torch.equal(before_actor, model.particle_adapter.weight)
    assert metrics["reward/anchor_accuracy"] == 0.0  # legacy actor-reward metric
    assert metrics["actor/reward_anchor_mean"] == 0.0
    assert metrics["q/correctness_anchor_accuracy"] == 1.0
    assert metrics["q/correctness_oracle_pass_at_k"] == 1.0
    assert metrics["q/correctness_actor_reward_disagreement_fraction"] == 0.75


@pytest.mark.parametrize(
    ("q_labels", "message"),
    [
        (torch.zeros(1, 3), "same \\[batch, K\\] shape"),
        (torch.tensor([[0.0, 1.0, float("nan"), 0.0]]), "finite"),
        (torch.tensor([[0.0, 1.0, 0.5, 0.0]]), "binary"),
    ],
)
def test_q_correctness_labels_are_strictly_validated(q_labels, message):
    model = DummyParticlePolicy()
    q_head = DummyQHead(model.hidden_size)
    engine = _engine(model)
    trainer = ParticleTrainer(
        model,
        engine,
        torch.optim.SGD(model.particle_adapter.parameters(), lr=0.05),
        torch.optim.SGD(q_head.parameters(), lr=0.1),
        q_head=q_head,
        config=TrainerConfig(kl_coefficient=0.0, injection_penalty_coefficient=0.0),
    )
    rollout = engine.generate([RolloutExample("What is one plus one?", "2", "dummy")])

    with pytest.raises(ValueError, match=message):
        trainer.train_rollout(replace(rollout, q_labels=q_labels))


@pytest.mark.integration
def test_rollout_rng_advances_but_fresh_engines_reproduce_first_draw():
    first_engine = _engine(DummyParticlePolicy(), seed=17, latent_sampler=None)
    examples = [RolloutExample("Compute 1+1", "2")]
    first = first_engine.generate(examples).particle_z.clone()
    second = first_engine.generate(examples).particle_z.clone()
    fresh = _engine(DummyParticlePolicy(), seed=17, latent_sampler=None).generate(examples).particle_z
    assert not torch.equal(first, second)
    assert torch.equal(first, fresh)


def test_matched_sampling_mode_does_not_force_branch_zero_greedy():
    """Ordinary K-sample evaluation must spend all four samples stochastically."""

    model = DummyParticlePolicy()
    engine = _engine(model, seed=123)
    engine.config = RolloutConfig(
        **{
            **engine.config.__dict__,
            "anchor_greedy": False,
            "compute_reference_logprobs": False,
        }
    )
    rollout = engine.generate([RolloutExample("Compute 1+1", "2")])
    # The branch remains exactly zero-latent, but its token now comes from the
    # same multinomial path as the other ordinary samples.  This assertion is
    # deliberately structural rather than seed-fragile.
    assert torch.equal(rollout.particle_z[:, 0], torch.zeros_like(rollout.particle_z[:, 0]))
    assert engine.config.anchor_greedy is False


def test_rollout_refuses_silent_context_overflow():
    model = DummyParticlePolicy()
    model.config = SimpleNamespace(max_position_embeddings=12)
    engine = _engine(model)
    with pytest.raises(ValueError, match="context window"):
        engine.generate([RolloutExample("This prompt is deliberately too long", "2")])


def test_untruncated_kl_logprobs_match_temperature_behavior_when_top_p_one():
    model = DummyParticlePolicy()
    engine = _engine(model)
    engine.config = RolloutConfig(
        **{**engine.config.__dict__, "temperature": 0.5, "compute_reference_logprobs": False}
    )
    rollout = engine.generate([RolloutExample("Compute 1+1", "2")])
    scored = score_actions(model, rollout)
    assert torch.allclose(scored.raw_logprobs, scored.logprobs)


@pytest.mark.integration
def test_shared_prefill_explicitly_masks_unperturbed_first_token_from_ppo():
    model = DummyParticlePolicy()
    engine = ParticleRolloutEngine(
        model,
        CharTokenizer(),
        config=RolloutConfig(
            k=4,
            latent_dim=2,
            max_new_tokens=2,
            response_prefix="ignored",
            first_token_mode="shared_prefill",
            use_cache=False,
            compute_reference_logprobs=False,
        ),
        latent_sampler=fixed_latents,
    )
    rollout = engine.generate([RolloutExample("Compute 1+1", "2")])
    assert rollout.generated_mask[:, :, 0].all()
    assert not rollout.action_mask[:, :, 0].any()


def test_checkpoint_roundtrip_restores_weights_optimizers_and_rollout_rng(tmp_path):
    from hrm_particle.checkpoint import load_checkpoint, save_checkpoint

    model = DummyParticlePolicy()
    q_head = DummyQHead(model.hidden_size)
    engine = _engine(model, seed=31, latent_sampler=None)
    actor_optimizer = torch.optim.Adam(model.particle_adapter.parameters(), lr=0.01)
    q_optimizer = torch.optim.Adam(q_head.parameters(), lr=0.01)
    examples = [RolloutExample("Compute 1+1", "2")]
    engine.generate(examples)  # advance persistent rollout RNG
    saved_actor = model.particle_adapter.weight.detach().clone()
    saved_q = q_head.linear.weight.detach().clone()
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        adapter=model.particle_adapter,
        q_head=q_head,
        step=7,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        rollout_engine=engine,
        config={"model": "dummy"},
        metadata={"loop_state": {"round_index": 1}},
    )
    expected_next_z = engine.generate(examples).particle_z
    with torch.no_grad():
        model.particle_adapter.weight.add_(100)
        q_head.linear.weight.sub_(100)
    restored_engine = _engine(model, seed=999, latent_sampler=None)
    result = load_checkpoint(
        path,
        adapter=model.particle_adapter,
        q_head=q_head,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        rollout_engine=restored_engine,
    )
    assert result["step"] == 7
    assert result["metadata"]["loop_state"]["round_index"] == 1
    assert torch.equal(model.particle_adapter.weight, saved_actor)
    assert torch.equal(q_head.linear.weight, saved_q)
    assert torch.equal(restored_engine.generate(examples).particle_z, expected_next_z)


def test_nonfinite_q_gradient_aborts_before_actor_step():
    class NaNQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, terminal_state, prompt_summary):
            return self.weight * torch.full(
                (terminal_state.shape[0],), float("nan"), device=terminal_state.device
            )

    model = DummyParticlePolicy()
    q_head = NaNQ()
    engine = _engine(model)
    actor_optimizer = torch.optim.SGD(model.particle_adapter.parameters(), lr=0.1)
    q_optimizer = torch.optim.SGD(q_head.parameters(), lr=0.1)
    trainer = ParticleTrainer(
        model,
        engine,
        actor_optimizer,
        q_optimizer,
        q_head=q_head,
        config=TrainerConfig(kl_coefficient=0.0, injection_penalty_coefficient=0.0),
    )
    rollout = engine.generate([RolloutExample("Compute 1+1", "2")])
    before = model.particle_adapter.weight.detach().clone()
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.train_rollout(rollout)
    assert torch.equal(before, model.particle_adapter.weight)


def test_new_run_refuses_occupied_output_before_loading_any_model(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("{}\n")
    config = {
        "optimization": {"actor_epochs_per_rollout": 1},
        "q_head": {"use_q_as_actor_reward": False},
        "generation": {"top_p": 1.0},
        "output": {"directory": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="occupied output directory"):
        train_from_config(config, output_dir=tmp_path)


def _write_records(directory):
    from hrm_particle.data import MathRecord

    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        records = [
            MathRecord(
                id=f"{split}-{index}",
                split=split,
                family="dummy",
                template_id="dummy_v1",
                prompt="What is one plus one?",
                answer="2",
                answer_type="integer",
                verifier="exact_fraction_v1",
                source="offline_test",
                difficulty=1,
                metadata={"semantic_signature": f"{split}-{index}"},
            )
            for index in range(2)
        ]
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def _entry_config(data_directory, output_directory):
    return {
        "seed": 9,
        "model": {
            "pretrained_model_name_or_path": "dummy",
            "revision": "offline",
            "dtype": "float32",
            "device": "cpu",
            "freeze_backbone": True,
        },
        "prompting": {"condition": "synth,cot"},
        "adapter": {"latent_size": 2, "bottleneck_size": 2},
        "particles": {"count": 4, "anchor_index": 0},
        "generation": {
            "max_new_tokens": 2,
            "explorer_temperature": 1.0,
            "top_p": 1.0,
            "first_token_mode": "causal_prefix",
            "response_prefix": "§",
            "use_cache_for_rollout": False,
        },
        "data": {
            "directory": str(data_directory),
            "train_file": "train.jsonl",
            "dev_file": "dev.jsonl",
            "max_train_examples": 2,
            "shuffle_train": True,
        },
        "optimization": {
            "actor_epochs_per_rollout": 1,
            "rollout_rounds": 1,
            "prompt_micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "adapter_learning_rate": 0.01,
            "kl_coefficient": 0.0,
            "injection_penalty_coefficient": 0.0,
        },
        "q_head": {
            "use_q_as_actor_reward": False,
            "learning_rate": 0.01,
            "ranking_weight": 0.1,
        },
        "checkpointing": {"save_every_updates": 1},
        "output": {"directory": str(output_directory)},
        "evaluation": {
            "splits": ["dev"],
            "prompt_batch_size": 1,
            "bootstrap_samples": 20,
            "save_candidate_text": True,
        },
    }


def _dummy_component_factory(_config):
    model = DummyParticlePolicy()
    # Match the real wrapper's public small-module names.
    model.adapter = model.particle_adapter
    model.q_head = DummyQHead(model.hidden_size)
    return model, CharTokenizer()


@pytest.mark.integration
def test_cli_facing_train_and_evaluate_entries_write_auditable_artifacts(
    tmp_path, monkeypatch
):
    import hrm_particle.trainer as trainer_module
    from hrm_particle.evaluate import evaluate_from_config

    data_directory = tmp_path / "data"
    run_directory = tmp_path / "run"
    eval_directory = tmp_path / "eval"
    _write_records(data_directory)
    config = _entry_config(data_directory, run_directory)
    monkeypatch.setattr(
        trainer_module, "build_components_from_config", _dummy_component_factory
    )
    result = train_from_config(config, output_dir=run_directory)
    checkpoint = run_directory / "checkpoint-last.pt"
    assert result["updates"] == 1
    assert checkpoint.is_file()
    assert (run_directory / "metrics.jsonl").is_file()
    assert (run_directory / "run-metadata.json").is_file()
    assert (run_directory / "resolved-config.json").is_file()

    # Exact resume sees a completed loop cursor and does not repeat examples.
    resumed = train_from_config(
        config,
        output_dir=run_directory,
        resume_from=checkpoint,
    )
    assert resumed["updates"] == result["updates"]
    assert (run_directory / "resume-events.jsonl").is_file()

    evaluation = evaluate_from_config(
        config,
        checkpoint=checkpoint,
        output_dir=eval_directory,
    )
    assert "dev" in evaluation["splits"]
    assert (eval_directory / "summary.json").is_file()
    audit_rows = [
        json.loads(line)
        for line in (eval_directory / "dev-per-prompt.jsonl").read_text().splitlines()
    ]
    assert audit_rows[0]["particle"]["response_texts"]
    assert audit_rows[0]["particle"]["rewards"]
    assert audit_rows[0]["particle"]["q_logits"]


def test_train_entry_rejects_dynamic_top_p_even_in_dry_run(tmp_path):
    config = _entry_config(tmp_path / "data", tmp_path / "run")
    config["generation"]["top_p"] = 0.95
    with pytest.raises(ValueError, match="top_p=1.0"):
        train_from_config(config, dry_run=True)


def test_evaluate_entry_requires_checkpoint_unless_explicit_step_zero(tmp_path):
    from hrm_particle.evaluate import evaluate_from_config

    config = _entry_config(tmp_path / "data", tmp_path / "run")
    with pytest.raises(ValueError, match="requires --checkpoint"):
        evaluate_from_config(config, checkpoint=None, output_dir=tmp_path / "eval")
