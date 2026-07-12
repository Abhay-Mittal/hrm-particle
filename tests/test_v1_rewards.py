from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest
import torch

import hrm_particle.v1_rewards as v1_rewards
from hrm_particle.rollout import ParticleRollout, RolloutExample
from hrm_particle.v1_rewards import (
    CandidateScore,
    MathVerifyScorer,
    OpenR1SandboxCodeScorer,
    RewardDependencyError,
    RewardRouter,
    STRICT_CODE_HARNESS_VERSION,
    STRICT_CODE_MAX_OUTPUT_BYTES,
    STRICT_CODE_TIMEOUT_SECONDS,
    build_strict_stdin_stdout_script,
    rescore_particle_rollout,
    run_remote_code_sandbox_canary,
)
from hrm_particle.verifier import VerificationResult


def _rollout(texts: list[list[str]]) -> ParticleRollout:
    b = len(texts)
    k = len(texts[0])
    sequence, time, hidden = 3, 1, 4
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
        rewards=torch.zeros(b, k),
        prompt_summary=torch.zeros(b, k, hidden),
        terminal_states=torch.zeros(b, k, hidden),
        response_texts=texts,
        example_ids=[f"example-{i}" for i in range(b)],
        references=["reference" for _ in range(b)],
        verification=[
            [VerificationResult(False, None, None, text, "reference", "unscored") for text in group]
            for group in texts
        ],
        temperature=0.8,
        top_p=1.0,
    )


def test_candidate_score_enforces_binary_and_bounded_reward():
    score = CandidateScore(correct=1, actor_reward=0.25, details={"source": "fixture"})
    assert score.correct is True
    assert score.actor_reward == 0.25
    with pytest.raises(ValueError, match="binary"):
        CandidateScore(correct=2, actor_reward=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CandidateScore(correct=False, actor_reward=float("nan"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CandidateScore(correct=False, actor_reward=1.01)


def test_math_verify_parses_gold_then_prediction_and_verifies_in_that_order():
    calls: list[tuple[Any, ...]] = []

    def parse(value):
        calls.append(("parse", value))
        return [f"parsed:{value}"]

    def verify(gold, prediction):
        calls.append(("verify", gold, prediction))
        return gold == ["parsed:42"] and prediction == ["parsed:answer"]

    score = MathVerifyScorer(parse_fn=parse, verify_fn=verify).score("answer", "42")
    assert score == CandidateScore(
        correct=True,
        actor_reward=1.0,
        details={"task": "math", "parseable": True},
    )
    assert calls == [
        ("parse", "42"),
        ("parse", "answer"),
        ("verify", ["parsed:42"], ["parsed:answer"]),
    ]


def test_math_verify_parse_failure_is_strictly_incorrect():
    score = MathVerifyScorer(
        parse_fn=lambda value: [] if value == "bad" else [value],
        verify_fn=lambda *_: pytest.fail("verify must not run for an empty parse"),
    ).score("bad", "gold")
    assert score.correct is False
    assert score.actor_reward == 0.0
    assert score.details["parseable"] is False


def test_math_verify_aborts_on_unparseable_gold() -> None:
    scorer = MathVerifyScorer(
        parse_fn=lambda value: [] if value == "bad-gold" else [value],
        verify_fn=lambda *_: True,
    )
    with pytest.raises(RuntimeError, match="gold answer"):
        scorer.score("candidate", "bad-gold")


def test_math_verify_missing_dependency_has_clear_error(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError("No module named 'math_verify'")

    monkeypatch.setattr(v1_rewards.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="math-verify"):
        MathVerifyScorer().score("prediction", "gold")


def test_open_r1_code_reward_is_batched_remote_and_separates_labels_from_shaping():
    captured = {}

    def fake_code_reward(**kwargs):
        captured.update(kwargs)
        return [0.99, 0.999999, 0.5, 1.0]

    infos = [
        {
            "language": "python",
            "test_cases": [
                {
                    "type": "stdin_stdout",
                    "fn_name": None,
                    "input": str(i),
                    "output": str(i),
                }
            ],
        }
        for i in range(4)
    ]
    scorer = OpenR1SandboxCodeScorer(
        provider="e2b",
        num_parallel=3,
        code_reward_fn=fake_code_reward,
        environ={"E2B_API_KEY": "fixture-key"},
    )
    scores = scorer.score_batch(["a", "b", "c", "d"], infos)

    assert captured["provider_type"] == "e2b"
    assert captured["num_parallel"] == 3
    assert captured["enforce_same_language"] is True
    assert captured["verification_info"] == infos
    assert captured["completions"] == [
        [{"role": "assistant", "content": value}] for value in ["a", "b", "c", "d"]
    ]
    # Only exact pass-all is correctness; even 0.999999 remains incorrect.
    assert [score.correct for score in scores] == [False, False, False, True]
    assert [score.actor_reward for score in scores] == pytest.approx([0.198, 0.1999998, 0.1, 1.0])


def test_code_scorer_can_disable_partial_actor_credit():
    scorer = OpenR1SandboxCodeScorer(
        provider="morph",
        shape_actor_reward=False,
        code_reward_fn=lambda **_: [0.75, 1.0],
        environ={"MORPH_API_KEY": "fixture-key"},
    )
    info = {"language": "python", "test_cases": []}
    scores = scorer.score_batch(["partial", "full"], [info, info])
    assert [score.correct for score in scores] == [False, True]
    assert [score.actor_reward for score in scores] == [0.0, 1.0]


def test_code_scorer_configurable_binary_reward_weight():
    scorer = OpenR1SandboxCodeScorer(
        binary_reward_weight=0.6,
        code_reward_fn=lambda **_: [0.5, 1.0],
        environ={"E2B_API_KEY": "fixture-key"},
    )
    info = {"language": "python", "test_cases": []}
    scores = scorer.score_batch(["partial", "full"], [info, info])
    assert [score.actor_reward for score in scores] == pytest.approx([0.2, 1.0])
    assert all(score.details["binary_reward_weight"] == 0.6 for score in scores)
    with pytest.raises(ValueError, match="binary_reward_weight"):
        OpenR1SandboxCodeScorer(binary_reward_weight=1.01)


def test_code_scorer_fails_before_call_when_remote_key_is_missing():
    called = False

    def must_not_run(**_):
        nonlocal called
        called = True
        return [1.0]

    scorer = OpenR1SandboxCodeScorer(provider="e2b", code_reward_fn=must_not_run, environ={})
    with pytest.raises(RuntimeError, match="E2B_API_KEY"):
        scorer.score_batch(["candidate"], [{"test_cases": []}])
    assert called is False


def test_code_scorer_missing_open_r1_dependency_has_clear_error(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError("No module named 'open_r1'")

    monkeypatch.setattr(v1_rewards.importlib, "import_module", missing)
    scorer = OpenR1SandboxCodeScorer(provider="e2b", environ={"E2B_API_KEY": "fixture-key"})
    with pytest.raises(RewardDependencyError, match="official Hugging Face `open-r1`"):
        scorer.score_batch(
            ["candidate"],
            [
                {
                    "language": "python",
                    "test_cases": [
                        {
                            "type": "stdin_stdout",
                            "fn_name": None,
                            "input": "",
                            "output": "",
                        }
                    ],
                }
            ],
        )


@pytest.mark.parametrize("bad_score", [None, float("nan"), 1.5, "not-a-number"])
def test_code_scorer_aborts_on_missing_or_invalid_remote_scores(bad_score):
    scorer = OpenR1SandboxCodeScorer(
        code_reward_fn=lambda **_: [bad_score],
        environ={"E2B_API_KEY": "fixture-key"},
    )
    info = {"test_cases": []}
    with pytest.raises(RuntimeError, match="abort/retry"):
        scorer.score_batch(["a"], [info])


def test_code_scorer_aborts_when_remote_provider_raises():
    def unavailable(**_):
        raise ConnectionError("fixture outage")

    scorer = OpenR1SandboxCodeScorer(
        code_reward_fn=unavailable,
        environ={"E2B_API_KEY": "fixture-key"},
    )
    with pytest.raises(RuntimeError, match="batch was aborted") as captured:
        scorer.score_batch(["a"], [{"test_cases": []}])
    assert isinstance(captured.value.__cause__, ConnectionError)


def test_strict_remote_harness_rejects_prefix_missing_extra_and_nonzero_exit() -> None:
    info = {
        "language": "python",
        "test_cases": [
            {
                "type": "stdin_stdout",
                "fn_name": None,
                "input": "ignored\n",
                "output": "one two\n",
            }
        ],
    }
    script = build_strict_stdin_stdout_script("print('one')", info)
    ast.parse(script)
    assert STRICT_CODE_HARNESS_VERSION in script
    assert f"TIMEOUT_SECONDS = {STRICT_CODE_TIMEOUT_SECONDS}" in script
    assert f"MAX_OUTPUT_BYTES = {STRICT_CODE_MAX_OUTPUT_BYTES}" in script
    assert "resource.RLIMIT_FSIZE" in script
    assert "process.returncode != 0" in script
    assert "os.killpg(process.pid, signal.SIGKILL)" in script
    assert 'tokenized_lines(actual) == tokenized_lines(case["output"])' in script
    assert "text.splitlines()" in script
    assert "zip(" not in script


def test_strict_harness_keeps_candidate_as_a_data_literal():
    candidate = '"""\nraise SystemExit(0)\n# \\ malicious-looking text'
    script = build_strict_stdin_stdout_script(
        candidate,
        {
            "language": "python",
            "test_cases": [
                {
                    "type": "stdin_stdout",
                    "fn_name": None,
                    "input": "",
                    "output": "",
                }
            ],
        },
    )
    tree = ast.parse(script)
    code_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CODE" for target in node.targets)
    )
    assert isinstance(code_assignment.value, ast.Constant)
    assert code_assignment.value.value == candidate


def test_default_code_path_uses_remote_provider_execute_scripts() -> None:
    captured = {}

    class Provider:
        def execute_scripts(self, scripts, languages):
            captured["scripts"] = scripts
            captured["languages"] = languages
            return [1.0, 1.0, 0.5]

    def factory(**kwargs):
        captured["factory"] = kwargs
        return Provider()

    info = {
        "language": "python",
        "test_cases": [{"type": "stdin_stdout", "fn_name": None, "input": "1\n", "output": "1\n"}],
    }
    scorer = OpenR1SandboxCodeScorer(
        provider_factory=factory,
        environ={"E2B_API_KEY": "fixture-key"},
    )
    scores = scorer.score_batch(["print(input())", "pass"], [info, info])
    assert captured["factory"] == {"provider_type": "e2b", "num_parallel": 8}
    assert captured["languages"] == ["python", "python", "python"]
    assert "hrm-v1-sandbox-ok" in captured["scripts"][0]
    assert all(STRICT_CODE_HARNESS_VERSION in script for script in captured["scripts"])
    assert [score.correct for score in scores] == [True, False]


def test_default_code_path_aborts_when_provider_collapses_outage_to_zeros() -> None:
    class Provider:
        def execute_scripts(self, scripts, languages):
            return [0.0] * len(scripts)

    info = {
        "language": "python",
        "test_cases": [{"type": "stdin_stdout", "fn_name": None, "input": "", "output": ""}],
    }
    scorer = OpenR1SandboxCodeScorer(
        provider_factory=lambda **_: Provider(),
        environ={"E2B_API_KEY": "fixture-key"},
    )
    with pytest.raises(RuntimeError, match="batch was aborted") as captured:
        scorer.score_batch(["pass"], [info])
    assert "known-good sentinel" in str(captured.value.__cause__)


def test_remote_sandbox_canary_exercises_strict_and_resource_cases() -> None:
    captured = {}

    class Provider:
        def execute_scripts(self, scripts, languages):
            captured["scripts"] = scripts
            captured["languages"] = languages
            return [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    scorer = OpenR1SandboxCodeScorer(
        provider_factory=lambda **_: Provider(),
        # A production canary still requires the selected remote credential.
        environ={"E2B_API_KEY": "fixture-key"},
        # The canary must bypass this injection and hit provider.execute_scripts.
        code_reward_fn=lambda **_: pytest.fail("injected code reward must not run"),
    )
    report = run_remote_code_sandbox_canary(scorer)
    assert report == {
        "known_good": 1.0,
        "changed_line_boundary": 0.0,
        "missing_output": 0.0,
        "extra_output": 0.0,
        "nonzero_exit": 0.0,
        "timeout": 0.0,
        "output_cap": 0.0,
    }
    assert captured["languages"] == ["python"] * 8
    assert len(captured["scripts"]) == 8
    assert all(STRICT_CODE_HARNESS_VERSION in script for script in captured["scripts"])
    assert str(STRICT_CODE_MAX_OUTPUT_BYTES + 1) in captured["scripts"][-1]


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_remote_sandbox_canary_aborts_on_missing_or_invalid_score(bad) -> None:
    class Provider:
        def execute_scripts(self, scripts, languages):
            return [1.0, 1.0, 0.0, bad, 0.0, 0.0]

    scorer = OpenR1SandboxCodeScorer(
        provider_factory=lambda **_: Provider(),
        environ={"E2B_API_KEY": "fixture-key"},
    )
    with pytest.raises(RuntimeError, match="canary"):
        run_remote_code_sandbox_canary(scorer, include_resource_checks=False)


def test_remote_sandbox_canary_rejects_semantically_wrong_provider() -> None:
    class Provider:
        def execute_scripts(self, scripts, languages):
            # A provider outage that is collapsed to all-zero rewards must not
            # silently label the actual training batch as incorrect.
            return [0.0] * len(scripts)

    scorer = OpenR1SandboxCodeScorer(
        provider_factory=lambda **_: Provider(),
        environ={"E2B_API_KEY": "fixture-key"},
    )
    with pytest.raises(RuntimeError, match="sandbox canary") as captured:
        run_remote_code_sandbox_canary(scorer, include_resource_checks=False)
    assert "known-good sentinel" in str(captured.value.__cause__)


def test_rescore_particle_rollout_batches_by_task_and_keeps_q_binary():
    texts = [["math-wrong", "math-right"], ["code-partial", "code-right"]]
    rollout = _rollout(texts)
    examples = [
        RolloutExample("math prompt", "math-gold", "math", {"task_type": "math"}),
        RolloutExample(
            "code prompt",
            "gold implementation",
            "code",
            {
                "task_type": "python",
                "verification_info": {"language": "python", "test_cases": []},
            },
        ),
    ]

    class FakeMath:
        def score_batch(self, predictions, golds):
            assert predictions == ["math-wrong", "math-right"]
            assert golds == ["math-gold", "math-gold"]
            return [
                CandidateScore(False, 0.0, {"task": "math"}),
                CandidateScore(True, 1.0, {"task": "math"}),
            ]

    class FakeCode:
        def score_batch(self, predictions, infos):
            assert predictions == ["code-partial", "code-right"]
            assert len(infos) == 2 and infos[0] is infos[1]
            return [
                CandidateScore(False, 0.1, {"task": "code", "fraction_passed": 0.5}),
                CandidateScore(True, 1.0, {"task": "code", "fraction_passed": 1.0}),
            ]

    router = RewardRouter(math_scorer=FakeMath(), code_scorer=FakeCode(), code_binary_weight=0.8)
    returned = rescore_particle_rollout(
        rollout,
        examples,
        router,
    )
    assert returned is rollout
    torch.testing.assert_close(rollout.rewards, torch.tensor([[0.0, 1.0], [0.1, 1.0]]))
    assert rollout.q_labels.tolist() == [[0.0, 1.0], [0.0, 1.0]]
    assert all(
        isinstance(item, VerificationResult) for group in rollout.verification for item in group
    )
    assert [[item.correct for item in group] for group in rollout.verification] == [
        [False, True],
        [False, True],
    ]
    assert rollout.verification[1][0].predicted_text == "code-partial"
    assert rollout.verification[1][0].expected_text == "gold implementation"


def test_k4_parallel_reward_alignment_and_binary_q_labels() -> None:
    texts = [
        ["m0", "m1", "m2", "m3"],
        ["c0", "c1", "c2", "c3"],
    ]
    rollout = _rollout(texts)
    info = {"language": "python", "test_cases": []}
    examples = [
        RolloutExample("math", "gold", "math", {"task_type": "math"}),
        RolloutExample(
            "code",
            "solution",
            "code",
            {"task_type": "code", "verification_info": info},
        ),
    ]

    class FakeMath:
        def score_batch(self, predictions, golds):
            assert predictions == texts[0]
            assert golds == ["gold"] * 4
            return [
                CandidateScore(False, 0.0),
                CandidateScore(True, 1.0),
                CandidateScore(False, 0.0),
                CandidateScore(True, 1.0),
            ]

    class FakeCode:
        def score_batch(self, predictions, infos):
            assert predictions == texts[1]
            assert infos == [info] * 4
            fractions = [0.0, 0.25, 1.0, 0.5]
            return [
                CandidateScore(
                    fraction == 1.0,
                    fraction,
                    {"fraction_passed": fraction},
                )
                for fraction in fractions
            ]

    router = RewardRouter(
        math_scorer=FakeMath(), code_scorer=FakeCode(), code_binary_weight=0.8
    )
    rescore_particle_rollout(rollout, examples, router)

    torch.testing.assert_close(
        rollout.rewards,
        torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.05, 1.0, 0.1]]),
    )
    assert rollout.q_labels.tolist() == [
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    assert [item.predicted_text for item in rollout.verification[0]] == texts[0]
    assert [item.predicted_text for item in rollout.verification[1]] == texts[1]


def test_rescore_requires_explicit_remote_code_scorer():
    rollout = _rollout([["candidate", "candidate-2"]])
    examples = [
        RolloutExample(
            "prompt",
            "gold",
            "code",
            {"task_type": "code", "verification_info": {"test_cases": []}},
        )
    ]
    with pytest.raises(ValueError, match="explicit OpenR1SandboxCodeScorer"):
        rescore_particle_rollout(rollout, examples)


def test_reward_router_has_no_local_code_execution_primitives():
    tree = ast.parse(inspect.getsource(v1_rewards))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    direct_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "subprocess" not in imported_roots
    assert direct_calls.isdisjoint({"eval", "exec", "compile"})
