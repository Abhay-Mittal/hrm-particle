from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import hrm_particle.v1_distributed as runner
from hrm_particle.gaussian import GaussianParticleAdapter, GaussianParticleConfig
from hrm_particle.rollout import ParticleRollout
from hrm_particle.v1_distributed import (
    _build_schedule,
    _clustered_bootstrap_delta,
    _strict_q_labels,
    validate_v1_config,
)


def _config() -> dict:
    import yaml

    path = Path(__file__).resolve().parents[1] / "configs" / "v1_3gpu_full.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _minimal_rollout() -> ParticleRollout:
    b, k, sequence, time, hidden = 1, 4, 2, 1, 3
    return ParticleRollout(
        model_input_ids=torch.zeros(b, k, sequence, dtype=torch.long),
        attention_mask=torch.ones(b, k, sequence, dtype=torch.bool),
        token_type_ids=torch.zeros(b, k, sequence, dtype=torch.long),
        position_ids=torch.zeros(b, k, sequence, dtype=torch.long),
        particle_mask=torch.ones(b, k, sequence, dtype=torch.bool),
        particle_z=torch.zeros(b, k, 2),
        action_ids=torch.zeros(b, k, time, dtype=torch.long),
        generated_mask=torch.ones(b, k, time, dtype=torch.bool),
        action_mask=torch.ones(b, k, time, dtype=torch.bool),
        action_positions=torch.zeros(b, k, time, dtype=torch.long),
        old_logprobs=torch.zeros(b, k, time),
        rewards=torch.tensor([[0.0, 0.2, 1.0, 0.0]]),
        prompt_summary=torch.zeros(b, k, hidden),
        terminal_states=torch.zeros(b, k, hidden),
        response_texts=[["a", "b", "c", "d"]],
        example_ids=["fixture"],
        references=["gold"],
        verification=[[]],
        temperature=0.8,
        top_p=1.0,
    )


def _noise_sweep_rows() -> list[dict[str, object]]:
    prompts = [
        ("math-a", "math"),
        ("math-b", "math"),
        ("code-a", "code"),
        ("code-b", "code"),
    ]
    labels = {
        0.0: (
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 1],
        ),
        0.01: (
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 1],
        ),
        0.02: (
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
        ),
    }
    rows: list[dict[str, object]] = []
    for scale, scale_labels in labels.items():
        for (prompt_id, task_type), correctness in zip(prompts, scale_labels):
            rows.append(
                {
                    "id": prompt_id,
                    "task_type": task_type,
                    "source": "fixture",
                    "scale": scale,
                    "correctness": list(correctness),
                    # Deliberately unrelated to correctness: shaped actor rewards
                    # are audit-only and must not influence scale selection.
                    "actor_rewards": [0.99, 0.01, 0.73, 0.41],
                    "response_texts": [
                        f"anchor-{prompt_id}",
                        f"scale-{scale}-one",
                        f"scale-{scale}-two",
                        f"scale-{scale}-three",
                    ],
                }
            )
    return list(reversed(rows))


def test_checked_in_v1_config_has_the_budgeted_invariants() -> None:
    summary = validate_v1_config(_config())
    assert summary["world_size"] == 3
    assert summary["k"] == 4
    assert summary["particle_mode"] == "gaussian"
    assert summary["q_candidates"] == 8192
    assert summary["bf16_model_and_trainables"] is True
    assert summary["trainable_particle_intervention"] is False
    assert summary["training_path"] == "fixed Gaussian scale selection plus Q-only SFT"


def test_two_gpu_poc_config_is_supported_without_a_gpu_sku_field() -> None:
    import yaml

    path = Path(__file__).resolve().parents[1] / "configs" / "v1_2gpu_poc.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    summary = validate_v1_config(config)
    assert summary["world_size"] == 2
    assert summary["q_candidates"] == 8192
    assert config["generation"]["train_max_new_tokens"] == 256
    assert "gpu_type" not in config["runtime"]
    assert config["rl"]["prompt_micro_batch_size"] == "auto"
    assert config["memory"]["target_fraction"] == 0.75


def test_learned_particle_mode_remains_an_explicit_optional_path() -> None:
    config = deepcopy(_config())
    config["particles"]["mode"] = "learned"

    summary = validate_v1_config(config)

    assert summary["particle_mode"] == "learned"
    assert summary["trainable_particle_intervention"] is True
    assert summary["training_path"] == "learned particle adapter plus Q and PPO"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_mode", "explicitly configured"),
        ("unknown_mode", "gaussian.*learned"),
        ("missing_artifact_path", "noise_scale_selection"),
        ("token_schedule", "response_fixed"),
        ("duplicate_scales", "sorted, unique"),
        ("zero_default", "declared positive candidate"),
        ("negative_tolerance", "finite and non-negative"),
    ],
)
def test_gaussian_config_invariants_fail_closed(case: str, message: str) -> None:
    config = deepcopy(_config())
    gaussian = config["particles"]["gaussian"]
    if case == "missing_mode":
        del config["particles"]["mode"]
    elif case == "unknown_mode":
        config["particles"]["mode"] = "mystery"
    elif case == "missing_artifact_path":
        config["paths"]["noise_scale_selection"] = ""
    elif case == "token_schedule":
        gaussian["schedule"] = "token_resampled"
    elif case == "duplicate_scales":
        gaussian["scale_candidates"] = [0.0, 0.01, 0.01]
    elif case == "zero_default":
        gaussian["default_relative_rms"] = 0.0
    elif case == "negative_tolerance":
        gaussian["oracle_tie_tolerance"] = -0.01
    else:  # pragma: no cover - protects the parametrized fixture itself
        raise AssertionError(case)

    with pytest.raises(ValueError, match=message):
        validate_v1_config(config)


def test_gaussian_engine_uses_intervention_width_and_fp32_latents() -> None:
    class FakePolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.adapter = GaussianParticleAdapter(
                GaussianParticleConfig(hidden_size=12, relative_rms_scale=0.02)
            )

    config = deepcopy(_config())
    policy = FakePolicy()
    engine = runner._engine(
        policy,
        tokenizer=object(),
        config=config,
        max_new_tokens=7,
        seed=31,
    )
    latents = engine.latent_sampler(
        2,
        4,
        engine.config.latent_dim,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(5),
    )

    assert engine.config.latent_dim == 12
    assert latents.shape == (2, 4, 12)
    assert latents.dtype == torch.float32
    assert torch.count_nonzero(latents[:, 0]) == 0
    explorer_rms = latents[:, 1:].square().mean(dim=-1).sqrt()
    torch.testing.assert_close(explorer_rms, torch.ones_like(explorer_rms))

    ordinary = runner._engine(
        policy,
        tokenizer=object(),
        config=config,
        max_new_tokens=7,
        seed=31,
        ordinary_sampling=True,
    )
    zero_latents = ordinary.latent_sampler(
        2,
        4,
        ordinary.config.latent_dim,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(5),
    )
    assert zero_latents.shape == (2, 4, 12)
    assert zero_latents.dtype == torch.bfloat16
    assert torch.count_nonzero(zero_latents) == 0


@pytest.mark.parametrize("world_size", [2, 3])
@pytest.mark.parametrize("micro_batch", [1, 2, 4, 8])
def test_dynamic_rank_microbatches_are_disjoint_and_exhaustive(
    world_size: int, micro_batch: int
) -> None:
    accumulation = 8 // micro_batch
    steps = 2
    total = steps * accumulation * micro_batch * world_size
    schedule = [{"id": f"row-{index}"} for index in range(total)]
    observed: list[str] = []
    for step in range(steps):
        for micro in range(accumulation):
            micro_ids: list[str] = []
            for rank in range(world_size):
                rows = runner._rank_microbatch_records(
                    schedule,
                    step=step,
                    micro=micro,
                    accumulation=accumulation,
                    prompt_micro_batch_size=micro_batch,
                    rank=rank,
                    world_size=world_size,
                )
                assert len(rows) == micro_batch
                micro_ids.extend(str(row["id"]) for row in rows)
            assert len(micro_ids) == len(set(micro_ids))
            observed.extend(micro_ids)
    assert observed == [str(row["id"]) for row in schedule]


def test_memory_probe_padding_covers_the_full_action_cap() -> None:
    rollout = _minimal_rollout()
    original_sequence = rollout.model_input_ids.shape[-1]
    padded = runner._pad_rollout_for_memory_probe(
        rollout, target_actions=4, filler_token_id=7
    )
    assert padded.action_ids.shape == (1, 4, 4)
    assert padded.model_input_ids.shape[-1] == original_sequence + 3
    assert padded.action_positions[0, 0].tolist() == [0, 1, 2, 3]
    assert padded.generated_mask[..., -3:].all()
    assert padded.action_mask[..., -3:].all()
    assert padded.particle_mask[..., -3:].all()
    assert (padded.model_input_ids[..., -3:] == 7).all()
    assert padded.reference_logprobs is None


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("particles", "count"), 5, "K=4"),
        (("particles", "top_p"), 0.95, "top_p=1.0"),
        (("model", "revision"), "main", "pin"),
        (("q_warmup", "target_candidate_examples"), 4096, "prompt groups times K"),
    ],
)
def test_v1_config_fails_closed_on_scientific_mismatches(path, value, message) -> None:
    config = deepcopy(_config())
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError, match=message):
        validate_v1_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("runtime", "mixed_precision"), "fp16", "bfloat16"),
        (("runtime", "dynamo_backend"), "inductor", "dynamo_backend"),
        (("memory", "auto_scale"), False, "auto_scale"),
        (("memory", "target_fraction"), 0.85, "target_fraction"),
        (("memory", "hard_limit_fraction"), 0.90, "hard_limit_fraction"),
        (("rl", "prompt_micro_batch_size"), 1, "must be auto"),
        (("rl", "gradient_accumulation_steps"), 8, "must be auto"),
        (("particles", "anchor_decode"), "sample", "greedy"),
        (("evaluation", "ordinary_sampling_candidates"), 3, "candidate counts"),
        (("verification", "fail_closed_without_sandbox"), False, "fail closed"),
        (("verification", "open_r1_commit"), "main", "full lowercase Git SHA"),
        (("evaluation", "evalplus_image"), "ganler/evalplus:latest", "SHA256 digest"),
    ],
)
def test_v1_config_enforces_runtime_and_evaluation_controls(path, value, message) -> None:
    config = deepcopy(_config())
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError, match=message):
        validate_v1_config(config)


def test_schedule_is_deterministic_and_has_exact_long_run_code_fraction() -> None:
    records = [
        *({"id": f"m-{i}", "task_type": "math"} for i in range(7)),
        *({"id": f"c-{i}", "task_type": "code"} for i in range(3)),
    ]
    first = _build_schedule(
        records, total_slots=100, code_fraction=0.2, code_enabled=True, seed=9
    )
    second = _build_schedule(
        records, total_slots=100, code_fraction=0.2, code_enabled=True, seed=9
    )
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert sum(item["task_type"] == "code" for item in first) == 20
    math_only = _build_schedule(
        records, total_slots=20, code_fraction=0.2, code_enabled=False, seed=9
    )
    assert all(item["task_type"] == "math" for item in math_only)


def test_noise_sweep_dev_sample_is_deterministic_unique_and_task_balanced() -> None:
    records = [
        *(
            {"id": f"math-{index}", "task_type": "math", "pool": "rl_train"}
            for index in range(20)
        ),
        *(
            {"id": f"code-{index}", "task_type": "code", "pool": "rl_train"}
            for index in range(10)
        ),
    ]

    first = runner._select_noise_sweep_records(
        records, prompt_groups=10, code_fraction=0.2, seed=47
    )
    second = runner._select_noise_sweep_records(
        records, prompt_groups=10, code_fraction=0.2, seed=47
    )
    selected_ids = [str(record["id"]) for record in first]

    assert selected_ids == [str(record["id"]) for record in second]
    assert len(first) == len(set(selected_ids)) == 10
    assert sum(record["task_type"] == "math" for record in first) == 8
    assert sum(record["task_type"] == "code" for record in first) == 2
    assert all(record["pool"] == "rl_train" for record in first)

    duplicate = [*records, dict(records[0])]
    with pytest.raises(ValueError, match="duplicate prompt IDs"):
        runner._select_noise_sweep_records(
            duplicate, prompt_groups=10, code_fraction=0.2, seed=47
        )


def test_noise_sweep_halves_are_disjoint_deterministic_and_task_balanced() -> None:
    records = [
        *(
            {
                "id": f"math-{index}",
                "task_type": "math",
                "source": f"math-source-{index % 3}",
            }
            for index in range(11)
        ),
        *(
            {
                "id": f"code-{index}",
                "task_type": "code",
                "source": f"code-source-{index % 2}",
            }
            for index in range(5)
        ),
    ]

    first = runner._split_noise_sweep_records(records, seed=53)
    second = runner._split_noise_sweep_records(records, seed=53)
    selection, confirmation = first

    assert [[row["id"] for row in half] for half in first] == [
        [row["id"] for row in half] for half in second
    ]
    selection_ids = {row["id"] for row in selection}
    confirmation_ids = {row["id"] for row in confirmation}
    assert len(selection) == len(confirmation) == 8
    assert selection_ids.isdisjoint(confirmation_ids)
    assert selection_ids | confirmation_ids == {row["id"] for row in records}
    for task_type in ("math", "code"):
        selection_count = sum(row["task_type"] == task_type for row in selection)
        confirmation_count = sum(row["task_type"] == task_type for row in confirmation)
        assert abs(selection_count - confirmation_count) <= 1


def test_noise_scale_metrics_are_paired_correctness_only_and_select_smallest_tie() -> None:
    aggregate = runner._aggregate_noise_scale_rows(
        _noise_sweep_rows(),
        candidates=[0.0, 0.01, 0.02],
        k=4,
        bootstrap_samples=200,
        seed=13,
    )
    baseline = aggregate["metrics"]["0.0"]
    improved = aggregate["metrics"]["0.01"]

    assert aggregate["prompt_ids"] == ["code-a", "code-b", "math-a", "math-b"]
    assert baseline["anchor_accuracy"] == pytest.approx(0.5)
    assert baseline["oracle_at_k"] == pytest.approx(0.75)
    assert baseline["explorer_mean_accuracy"] == pytest.approx(1.0 / 3.0)
    assert baseline["mixed_fraction"] == pytest.approx(0.5)
    assert baseline["rescue_fraction"] == pytest.approx(0.25)
    assert improved["oracle_at_k"] == pytest.approx(1.0)
    assert improved["oracle_delta_vs_zero"]["mean_delta"] == pytest.approx(0.25)
    assert improved["by_task"]["math"]["prompt_groups"] == 2
    assert improved["by_task"]["code"]["prompt_groups"] == 2
    assert (
        runner._select_noise_scale(
            aggregate["metrics"],
            candidates=[0.0, 0.01, 0.02],
            tie_tolerance=0.0,
        )
        == 0.01
    )

    worse = deepcopy(aggregate["metrics"])
    worse["0.0"]["oracle_at_k"] = 1.0
    worse["0.01"]["oracle_at_k"] = 0.75
    worse["0.02"]["oracle_at_k"] = 0.5
    with pytest.raises(RuntimeError, match="refusing to force a harmful particle"):
        runner._select_noise_scale(
            worse,
            candidates=[0.0, 0.01, 0.02],
            tie_tolerance=0.0,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("nonbinary", "binary"),
        ("duplicate", "duplicate id/scale"),
        ("missing_cell", "incomplete"),
        ("anchor_correctness", "branch-zero correctness"),
        ("anchor_text", "branch-zero text"),
    ],
)
def test_noise_scale_aggregation_rejects_unpaired_or_invalid_rows(
    case: str, message: str
) -> None:
    rows = _noise_sweep_rows()
    if case == "nonbinary":
        rows[0]["correctness"] = [1, 0.5, 0, 0]
    elif case == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif case == "missing_cell":
        rows.pop()
    elif case == "anchor_correctness":
        target = next(row for row in rows if row["scale"] == 0.01 and row["id"] == "math-a")
        target["correctness"] = [1, 1, 0, 0]
    elif case == "anchor_text":
        target = next(row for row in rows if row["scale"] == 0.01 and row["id"] == "math-a")
        target["response_texts"] = ["changed-anchor", "one", "two", "three"]
    else:  # pragma: no cover - protects the parametrized fixture itself
        raise AssertionError(case)

    with pytest.raises(RuntimeError, match=message):
        runner._aggregate_noise_scale_rows(
            rows,
            candidates=[0.0, 0.01, 0.02],
            k=4,
            bootstrap_samples=100,
            seed=13,
        )


def test_q_labels_are_strict_correctness_even_when_actor_rewards_are_shaped() -> None:
    rollout = _minimal_rollout()
    rollout.q_labels = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    assert torch.equal(_strict_q_labels(rollout), rollout.q_labels)
    rollout.q_labels[0, 1] = 0.25
    with pytest.raises(ValueError, match="binary"):
        _strict_q_labels(rollout)


def test_symbolic_bootstrap_resamples_templates_not_correlated_instances() -> None:
    result = _clustered_bootstrap_delta(
        system=[1, 1, 1, 0, 0, 0],
        baseline=[0, 0, 0, 1, 1, 1],
        clusters=["template-a"] * 3 + ["template-b"] * 3,
        samples=500,
        seed=4,
    )
    assert result["clusters"] == 2.0
    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["low"] <= -0.9
    assert result["high"] >= 0.9


def test_math_majority_clusters_equivalent_forms_and_ties_by_earliest() -> None:
    values = {"one-half": 0.5, "decimal": 0.5, "two-a": 2.0, "two-b": 2.0}

    def parse(text):
        return [] if text == "bad" else [values[text]]

    def verify(gold, prediction):
        return gold[0] == prediction[0]

    index, normalized = runner._math_equivalence_majority_vote(
        ["one-half", "decimal", "two-a", "two-b"],
        parse_fn=parse,
        verify_fn=verify,
    )
    assert index == 0
    assert normalized == ["0.5", "0.5", "2.0", "2.0"]
    assert runner._math_equivalence_majority_vote(
        ["bad"], parse_fn=parse, verify_fn=verify
    )[0] == 0


def test_final_q_calibration_is_task_specific_and_uses_disjoint_safety_rows() -> None:
    config = deepcopy(_config())
    config["q_warmup"]["bootstrap_samples"] = 100
    config["q_warmup"]["minimum_math_calibration_prompts"] = 20
    config["q_warmup"]["minimum_code_calibration_prompts"] = 20
    rows = []
    for task_type in ("math", "code"):
        for split in ("margin_select", "safety_test"):
            for index in range(30):
                # Q always prefers branch one. That is a safe rescue for math,
                # but deliberately harmful on the untouched code safety split.
                labels = [0, 1, 0, 0]
                if task_type == "code" and split == "safety_test":
                    labels = [1, 0, 0, 0]
                rows.append(
                    {
                        "id": f"{task_type}-{split}-{index}",
                        "task_type": task_type,
                        "split": split,
                        "logits": [0.0, 1.0, -1.0, -1.0],
                        "correctness": labels,
                    }
                )

    gates = runner._calibrate_final_q_rows(rows, config)

    assert gates["math"]["ready"] is True
    assert gates["code"]["selection"]["ready"] is True
    assert gates["code"]["safety_test"]["ready"] is False
    assert gates["code"]["ready"] is False


def test_missing_final_q_calibration_forces_anchor_without_using_q(tmp_path: Path) -> None:
    config = deepcopy(_config())
    config["paths"]["final_q_calibration"] = str(tmp_path / "missing.json")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")

    gate = runner._load_final_q_gate(config, checkpoint, task_type="math")

    assert gate["ready"] is False
    assert "branch zero forced" in gate["reason"]


def test_metrics_reconciliation_is_idempotent_at_the_checkpoint_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 1, "elapsed_seconds": 3.0, "loss": 4.0},
        {"step": 2, "elapsed_seconds": 7.0, "loss": 3.0},
        {"step": 3, "elapsed_seconds": 11.0, "loss": 2.0},
        {"step": 2.0, "elapsed_seconds": 8.0, "loss": 1.0},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    elapsed = runner._reconcile_metrics_log(path, checkpoint_step=2)
    reconciled = [json.loads(line) for line in path.read_text().splitlines()]

    assert elapsed == 8.0
    assert [row["step"] for row in reconciled] == [1, 2.0]
    assert reconciled[-1]["loss"] == 1.0
    assert runner._reconcile_metrics_log(path, checkpoint_step=2) == 8.0


def test_checkpoint_header_rejects_pilot_as_a_full_training_resume() -> None:
    payload = {
        "format": runner.CHECKPOINT_FORMAT,
        "version": runner.CHECKPOINT_VERSION,
        "stage": "pilot",
    }
    runner._validate_checkpoint_header(payload, expected_stage="pilot")
    with pytest.raises(RuntimeError, match="cannot be used as 'train'"):
        runner._validate_checkpoint_header(payload, expected_stage="train")


@pytest.mark.parametrize("pilot", [False, True])
def test_gaussian_rl_is_rejected_before_distributed_initialization(
    pilot: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_distributed_context",
        lambda *_args, **_kwargs: pytest.fail(
            "Gaussian RL rejection must happen before process-group initialization"
        ),
    )

    with pytest.raises(ValueError, match="parameter-free and Q-only"):
        runner.stage_rl(_config(), pilot=pilot, resume_from=None)


def test_gaussian_eval_resolves_only_the_configured_q_checkpoint(tmp_path: Path) -> None:
    config = deepcopy(_config())
    q_checkpoint = tmp_path / "q-head.safetensors"
    q_checkpoint.write_bytes(b"fixture-q")
    config["paths"]["q_checkpoint"] = str(q_checkpoint)

    assert runner._resolve_eval_checkpoint(config, None) == q_checkpoint.resolve()
    assert (
        runner._resolve_eval_checkpoint(config, q_checkpoint)
        == q_checkpoint.resolve()
    )
    with pytest.raises(ValueError, match="only the configured q_checkpoint"):
        runner._resolve_eval_checkpoint(config, tmp_path / "train-checkpoint.pt")

    q_checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="run q_warmup first"):
        runner._resolve_eval_checkpoint(config, None)


def test_process_group_cleanup_never_enters_a_barrier(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runner.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        runner.dist,
        "barrier",
        lambda: pytest.fail("cleanup must not enter a collective"),
    )
    monkeypatch.setattr(runner.dist, "destroy_process_group", lambda: calls.append("destroy"))

    runner._destroy_process_group()

    assert calls == ["destroy"]


def test_data_fingerprints_hash_the_consumed_files_not_only_the_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    files: dict[str, dict[str, object]] = {}
    for name in ("q_warm", "rl_train", "eval"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"id": name}) + "\n", encoding="utf-8")
        files[name] = {
            "path": path.name,
            "sha256": runner.sha256_file(path),
            "records": 1,
        }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator_version": "fixture",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "validate_v1_directory",
        lambda _directory: {"manifest_sha256": runner.sha256_file(manifest_path)},
    )
    config = deepcopy(_config())
    config["paths"]["data_directory"] = str(tmp_path)

    fingerprint = runner._validated_data_fingerprints(config)

    assert fingerprint["manifest_sha256"] == runner.sha256_file(manifest_path)
    assert fingerprint["files"]["rl_train"]["sha256"] == files["rl_train"]["sha256"]
    (tmp_path / "rl_train.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        runner._validated_data_fingerprints(config)


def test_initial_adapter_artifact_round_trip_and_checksum(
    tmp_path: Path, monkeypatch
) -> None:
    config = deepcopy(_config())
    config["paths"]["q_state_directory"] = str(tmp_path)
    monkeypatch.setattr(runner, "_initial_adapter_identity", lambda _config: {"fixture": 1})
    adapter = nn.Linear(4, 3).to(dtype=torch.bfloat16)
    context = runner.DistributedContext(0, 0, 1, torch.device("cpu"))

    metadata = runner._save_initial_adapter_artifact(adapter, config, context)
    expected = {
        name: value.detach().clone() for name, value in adapter.state_dict().items()
    }
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.add_(1)
    loaded = runner._load_initial_adapter_artifact(adapter, config)

    assert loaded["sha256"] == metadata["sha256"]
    assert all(
        torch.equal(adapter.state_dict()[name], expected_value)
        for name, expected_value in expected.items()
    )

    _, metadata_path = runner._initial_adapter_paths(config)
    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        runner._load_initial_adapter_artifact(adapter, config)


def test_parameterless_gaussian_artifact_round_trip_and_buffer_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = deepcopy(_config())
    config["paths"]["q_state_directory"] = str(tmp_path)
    monkeypatch.setattr(runner, "_initial_adapter_identity", lambda _config: {"fixture": 2})
    intervention = GaussianParticleAdapter(
        GaussianParticleConfig(hidden_size=8, relative_rms_scale=0.02)
    )
    context = runner.DistributedContext(0, 0, 1, torch.device("cpu"))

    assert list(intervention.parameters()) == []
    assert runner._module_device(intervention) == torch.device("cpu")
    before = runner._parameter_checksum(intervention)
    metadata = runner._save_initial_adapter_artifact(intervention, config, context)

    intervention.set_relative_rms_scale(0.05)
    changed = runner._parameter_checksum(intervention)
    assert changed.device.type == "cpu"
    assert float(changed) != float(before)

    loaded = runner._load_initial_adapter_artifact(intervention, config)

    assert loaded["sha256"] == metadata["sha256"]
    assert intervention.relative_rms_scale.dtype == torch.float32
    assert float(intervention.relative_rms_scale) == pytest.approx(0.02)
    assert float(runner._parameter_checksum(intervention)) == pytest.approx(float(before))


def test_rl_q_objective_receives_fp32_logits_and_labels(monkeypatch) -> None:
    class TinyQHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(6, 1).to(dtype=torch.bfloat16)

        def forward(self, terminal: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
            inputs = torch.cat((terminal, prompt), dim=-1).to(torch.bfloat16)
            return self.linear(inputs).squeeze(-1)

    class TinyPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = nn.Linear(1, 1).to(dtype=torch.bfloat16)
            self.q_head = TinyQHead()

    rollout = _minimal_rollout()
    rollout.q_labels = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    monkeypatch.setattr(
        runner,
        "score_actions",
        lambda _policy, batch: SimpleNamespace(
            logprobs=batch.old_logprobs,
            raw_logprobs=batch.old_logprobs,
            output=SimpleNamespace(relative_rms=None),
        ),
    )
    observed: dict[str, torch.dtype] = {}
    original_loss = runner.supervised_q_loss

    def recording_q_loss(logits, labels, *, ranking_weight):
        observed["logits"] = logits.dtype
        observed["labels"] = labels.dtype
        return original_loss(logits, labels, ranking_weight=ranking_weight)

    monkeypatch.setattr(runner, "supervised_q_loss", recording_q_loss)

    output = runner.RLTrainModule(TinyPolicy(), _config())(rollout)

    assert torch.isfinite(output["loss"])
    assert observed == {"logits": torch.float32, "labels": torch.float32}


def test_q_scores_cannot_change_actor_loss(monkeypatch) -> None:
    class ConstantQ(nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(value, dtype=torch.bfloat16))

        def forward(self, terminal: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
            return self.bias.expand(terminal.shape[0])

    class TinyPolicy(nn.Module):
        def __init__(self, q_value: float) -> None:
            super().__init__()
            self.adapter = nn.Linear(1, 1).to(dtype=torch.bfloat16)
            self.q_head = ConstantQ(q_value)

    rollout = _minimal_rollout()
    rollout.rewards = torch.tensor([[0.0, 0.1, 1.0, 0.4]])
    rollout.q_labels = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    monkeypatch.setattr(
        runner,
        "score_actions",
        lambda _policy, batch: SimpleNamespace(
            logprobs=batch.old_logprobs + 0.05,
            raw_logprobs=batch.old_logprobs + 0.05,
            output=SimpleNamespace(relative_rms=None),
        ),
    )

    low_q = runner.RLTrainModule(TinyPolicy(-3.0), _config())(rollout)
    high_q = runner.RLTrainModule(TinyPolicy(3.0), _config())(rollout)
    rollout.q_labels = torch.zeros_like(rollout.q_labels)
    different_labels = runner.RLTrainModule(TinyPolicy(3.0), _config())(rollout)

    torch.testing.assert_close(low_q["actor_loss"], high_q["actor_loss"])
    torch.testing.assert_close(high_q["actor_loss"], different_labels["actor_loss"])
    assert not torch.equal(low_q["q_loss"], high_q["q_loss"])
    assert not torch.equal(high_q["q_loss"], different_labels["q_loss"])
