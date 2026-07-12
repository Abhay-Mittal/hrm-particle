"""Verifier reward routing for the mixed math/Python V1 experiment.

The learned Q head is an outcome predictor, so its target is always strict
binary correctness.  The actor may additionally receive a small amount of
partial test credit for code.  These two signals are deliberately materialized
as different tensors by :func:`rescore_particle_rollout`.

Generated Python is never run by this module.  Code scoring is delegated only
to Open-R1's official ``code_reward`` with an explicitly configured E2B or
Morph remote sandbox.  Missing credentials or dependencies stop the batch
instead of silently falling back to local execution.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from .rollout import ParticleRollout, RolloutExample
from .verifier import VerificationResult
from .v1_utils import extract_python_code


_CODE_PROVIDERS = {"e2b": "E2B_API_KEY", "morph": "MORPH_API_KEY"}
_CODE_TASK_NAMES = {"code", "coding", "python", "python_code"}
_MATH_TASK_NAMES = {"math", "mathematics", "arithmetic"}


@dataclass(frozen=True)
class CandidateScore:
    """One externally verified candidate.

    ``correct`` is the strict target for Q.  ``actor_reward`` may be shaped but
    must remain finite and in ``[0, 1]``.  ``details`` contains small audit
    metadata only; code, tests, and credentials are intentionally not copied
    into it.
    """

    correct: bool
    actor_reward: float
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        correct = self.correct
        if isinstance(correct, bool):
            normalized_correct = correct
        elif isinstance(correct, Real) and float(correct) in {0.0, 1.0}:
            normalized_correct = bool(correct)
        else:
            raise ValueError("CandidateScore.correct must be binary (False/True or 0/1)")

        reward = float(self.actor_reward)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("CandidateScore.actor_reward must be finite and in [0, 1]")
        if not isinstance(self.details, Mapping):
            raise TypeError("CandidateScore.details must be a mapping")

        object.__setattr__(self, "correct", normalized_correct)
        object.__setattr__(self, "actor_reward", reward)
        object.__setattr__(self, "details", dict(self.details))


class MathVerifyScorer:
    """Strict math correctness backed by ``math_verify``.

    Imports are lazy so data preparation and code-only jobs do not require the
    math extra.  ``parse_fn`` and ``verify_fn`` are dependency-injection hooks
    used by tests; production callers should leave both unset.
    """

    def __init__(
        self,
        *,
        parse_fn: Optional[Callable[[object], Any]] = None,
        verify_fn: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self._parse_fn = parse_fn
        self._verify_fn = verify_fn

    def _functions(self) -> tuple[Callable[[object], Any], Callable[[Any, Any], Any]]:
        if self._parse_fn is not None and self._verify_fn is not None:
            return self._parse_fn, self._verify_fn
        try:
            module = importlib.import_module("math_verify")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Math reward requires `math-verify`; install the V1 math reward "
                "dependencies before training."
            ) from exc

        parse_fn = self._parse_fn or getattr(module, "parse", None)
        verify_fn = self._verify_fn or getattr(module, "verify", None)
        if not callable(parse_fn) or not callable(verify_fn):
            raise RuntimeError("`math_verify` must expose callable parse() and verify()")
        self._parse_fn = parse_fn
        self._verify_fn = verify_fn
        return parse_fn, verify_fn

    def score(self, prediction: object, gold: object) -> CandidateScore:
        parse_fn, verify_fn = self._functions()
        # A broken/unparseable reference is a data or verifier failure, not
        # evidence that every sampled answer is wrong. Abort the batch rather
        # than poisoning Q labels with false negatives.
        try:
            gold_parsed = parse_fn(gold)
        except Exception as exc:
            raise RuntimeError("math verifier failed while parsing the gold answer") from exc
        if not gold_parsed:
            raise RuntimeError("math gold answer is not parseable; abort/review the dataset row")
        try:
            # This order is intentional.  math_verify.verify expects the gold
            # parse first and the model prediction second.
            prediction_parsed = parse_fn(prediction)
            parseable = bool(prediction_parsed)
            correct = bool(verify_fn(gold_parsed, prediction_parsed)) if parseable else False
        except Exception as exc:
            return CandidateScore(
                correct=False,
                actor_reward=0.0,
                details={"task": "math", "error": f"math verification failed: {exc}"},
            )
        return CandidateScore(
            correct=correct,
            actor_reward=float(correct),
            details={"task": "math", "parseable": parseable},
        )

    def score_batch(
        self, predictions: Sequence[object], golds: Sequence[object]
    ) -> list[CandidateScore]:
        if len(predictions) != len(golds):
            raise ValueError("math predictions and gold answers must have equal lengths")
        return [self.score(prediction, gold) for prediction, gold in zip(predictions, golds)]


STRICT_CODE_HARNESS_VERSION = "strict-stdin-stdout-v1"
STRICT_CODE_TIMEOUT_SECONDS = 5
STRICT_CODE_MAX_OUTPUT_BYTES = 64 * 1024


class RewardDependencyError(RuntimeError):
    """Raised when an explicitly configured verifier dependency is unavailable."""


def build_strict_stdin_stdout_script(prediction: str, verification_info: Mapping[str, Any]) -> str:
    """Build a remote-only checker with exact token counts and return-code checks.

    The upstream Open-R1 reward script historically compared lines with
    ``zip``, which can accept a correct prefix with missing output.  V1 vendors
    this small strict checker while still using Open-R1's E2B/Morph provider.
    Candidate code remains data inside this string and is executed only by the
    remote sandbox.
    """

    if not isinstance(verification_info, Mapping):
        raise ValueError("verification_info must be a mapping")
    if str(verification_info.get("language", "")).casefold() != "python":
        raise ValueError("strict code verifier accepts Python only")
    cases = verification_info.get("test_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("strict code verifier requires non-empty test_cases")
    clean_cases: list[dict[str, str]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or case.get("type") != "stdin_stdout":
            raise ValueError(f"test case {index} must have type stdin_stdout")
        if case.get("fn_name") not in (None, ""):
            raise ValueError(f"test case {index} cannot define fn_name")
        if not isinstance(case.get("input"), str) or not isinstance(case.get("output"), str):
            raise ValueError(f"test case {index} input/output must be strings")
        clean_cases.append({"input": case["input"], "output": case["output"]})

    code = extract_python_code(str(prediction))
    # JSON literals safely preserve arbitrary candidate/test text without
    # interpolating it as verifier source code.
    code_literal = json.dumps(code)
    cases_literal = json.dumps(json.dumps(clean_cases, separators=(",", ":")))
    return f"""# {STRICT_CODE_HARNESS_VERSION}
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile

CODE = {code_literal}
TEST_CASES = json.loads({cases_literal})
TIMEOUT_SECONDS = {STRICT_CODE_TIMEOUT_SECONDS}
MAX_OUTPUT_BYTES = {STRICT_CODE_MAX_OUTPUT_BYTES}

def tokenized_lines(text):
    # Preserve line cardinality while tolerating whitespace between tokens.
    return [line.split() for line in text.splitlines()]

def limit_child_output():
    # stdout/stderr are regular temporary files. RLIMIT_FSIZE stops the child
    # while it is writing, rather than buffering unbounded output in a pipe.
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)

def strict_evaluate(code, test_cases):
    passed = 0
    with tempfile.TemporaryDirectory() as directory:
        path = directory + "/candidate.py"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)
        for case_index, case in enumerate(test_cases):
            stdout_path = directory + f"/stdout-{{case_index}}.txt"
            stderr_path = directory + f"/stderr-{{case_index}}.txt"
            with open(stdout_path, "wb") as stdout_handle, open(
                stderr_path, "wb"
            ) as stderr_handle:
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-I", path],
                        stdin=subprocess.PIPE,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                        cwd=directory,
                        start_new_session=True,
                        preexec_fn=limit_child_output,
                    )
                except OSError:
                    continue
                try:
                    process.communicate(input=case["input"], timeout=TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    # Kill the entire new session, including descendants.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    continue
                # The direct candidate may have spawned background children.
                # End its isolated process group before inspecting output so a
                # descendant cannot keep writing after the candidate returns.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            if process.returncode != 0:
                continue
            if (
                os.path.getsize(stdout_path) > MAX_OUTPUT_BYTES
                or os.path.getsize(stderr_path) > MAX_OUTPUT_BYTES
            ):
                continue
            with open(stdout_path, "r", encoding="utf-8", errors="replace") as handle:
                actual = handle.read(MAX_OUTPUT_BYTES + 1)
            # Both line count and per-line token count/order must match. A
            # correct prefix, an extra token, or a line-boundary change fails.
            if tokenized_lines(actual) == tokenized_lines(case["output"]):
                passed += 1
    return passed / len(test_cases)

strict_evaluate(CODE, TEST_CASES)
"""


class OpenR1SandboxCodeScorer:
    """Batch Python scoring through Open-R1 and a remote E2B/Morph sandbox.

    The strict remote harness returns the fraction of hidden tests passed. Q
    receives exact pass-all (``fraction == 1``) as its binary target. When shaping is enabled, the
    actor receives ``0.8 * binary + 0.2 * fraction``; disabling shaping gives
    it the same strict binary reward as Q.
    """

    def __init__(
        self,
        *,
        provider: str = "e2b",
        num_parallel: int = 8,
        shape_actor_reward: bool = True,
        binary_reward_weight: float = 0.8,
        code_reward_fn: Optional[Callable[..., Sequence[Optional[float]]]] = None,
        provider_factory: Optional[Callable[..., Any]] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        provider = provider.lower().strip()
        if provider not in _CODE_PROVIDERS:
            raise ValueError("code provider must be exactly 'e2b' or 'morph'")
        if num_parallel <= 0:
            raise ValueError("num_parallel must be positive")
        if not 0.0 <= binary_reward_weight <= 1.0:
            raise ValueError("binary_reward_weight must lie in [0, 1]")
        self.provider = provider
        self.num_parallel = int(num_parallel)
        self.shape_actor_reward = bool(shape_actor_reward)
        self.binary_reward_weight = float(binary_reward_weight)
        self._provider_factory = provider_factory
        self._code_reward_fn = code_reward_fn
        self._environ = environ

    def _require_remote_credentials(self) -> None:
        key_name = _CODE_PROVIDERS[self.provider]
        environ = os.environ if self._environ is None else self._environ
        if not str(environ.get(key_name, "")).strip():
            raise RuntimeError(
                f"Remote code reward is disabled: set {key_name} for the "
                f"{self.provider} sandbox provider. No local fallback is allowed."
            )

    def _provider(self) -> Any:
        if self._provider_factory is not None:
            return self._provider_factory(
                provider_type=self.provider, num_parallel=self.num_parallel
            )
        try:
            module = importlib.import_module("open_r1.rewards")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RewardDependencyError(
                "Code reward requires the official Hugging Face `open-r1` package "
                "and the selected remote sandbox dependency."
            ) from exc
        factory = getattr(module, "get_provider", None)
        if not callable(factory):
            raise RewardDependencyError("official `open_r1.rewards` does not expose get_provider()")
        return factory(provider_type=self.provider, num_parallel=self.num_parallel)

    def _strict_remote_rewards(
        self,
        predictions: Sequence[str],
        verification_info: Sequence[Mapping[str, Any]],
    ) -> Sequence[Optional[float]]:
        scripts = [
            build_strict_stdin_stdout_script(prediction, info)
            for prediction, info in zip(predictions, verification_info)
        ]
        # Open-R1 providers may catch a transport failure and return an all-zero
        # vector. A known-good sentinel in every call makes that distinguishable
        # from a genuinely all-wrong model batch.
        sentinel = build_strict_stdin_stdout_script(
            'print("hrm-v1-sandbox-ok")',
            {
                "language": "python",
                "test_cases": [
                    {
                        "type": "stdin_stdout",
                        "fn_name": None,
                        "input": "",
                        "output": "hrm-v1-sandbox-ok\n",
                    }
                ],
            },
        )
        provider = self._provider()
        execute = getattr(provider, "execute_scripts", None)
        if not callable(execute):
            raise RuntimeError("Open-R1 provider does not expose execute_scripts()")
        raw = execute([sentinel, *scripts], ["python"] * (len(scripts) + 1))
        if raw is None:
            raise RuntimeError("remote provider returned no sentinel result")
        try:
            values = list(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("remote provider returned malformed sentinel results") from exc
        if len(values) != len(scripts) + 1:
            raise RuntimeError("remote provider returned the wrong sentinel batch length")
        try:
            sentinel_score = float(values[0])
        except (TypeError, ValueError):
            sentinel_score = float("nan")
        if not math.isfinite(sentinel_score) or not math.isclose(
            sentinel_score, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                "remote provider failed the per-batch known-good sentinel; abort/retry"
            )
        return values[1:]

    def score_batch(
        self,
        predictions: Sequence[str],
        verification_info: Sequence[Mapping[str, Any]],
    ) -> list[CandidateScore]:
        if len(predictions) != len(verification_info):
            raise ValueError("code predictions and verification_info must have equal lengths")
        if not predictions:
            return []
        for index, info in enumerate(verification_info):
            if not isinstance(info, Mapping) or "test_cases" not in info:
                raise ValueError(
                    f"verification_info[{index}] must be a mapping containing test_cases"
                )

        # Check configuration before loading or calling any evaluator.  There
        # is intentionally no alternate path that could run a candidate here.
        self._require_remote_credentials()
        try:
            if self._code_reward_fn is not None:
                # Dependency-injection compatibility for deterministic unit
                # tests. Production must leave this unset and uses the strict
                # provider script above.
                raw_rewards = self._code_reward_fn(
                    completions=[
                        [{"role": "assistant", "content": str(prediction)}]
                        for prediction in predictions
                    ],
                    verification_info=list(verification_info),
                    num_parallel=self.num_parallel,
                    provider_type=self.provider,
                    enforce_same_language=True,
                )
            else:
                raw_rewards = self._strict_remote_rewards(predictions, verification_info)
        except RewardDependencyError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Open-R1 remote {self.provider} code verification failed; "
                "the batch was aborted without assigning reward."
            ) from exc
        if raw_rewards is None:
            raise RuntimeError(
                "remote verifier returned no batch; abort/retry instead of assigning labels"
            )
        try:
            raw_rewards = list(raw_rewards)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "remote verifier returned a malformed batch; abort/retry instead of "
                "assigning labels"
            ) from exc
        if len(raw_rewards) != len(predictions):
            raise RuntimeError("Open-R1 code_reward returned the wrong number of scores")

        scores: list[CandidateScore] = []
        for index, raw_reward in enumerate(raw_rewards):
            if raw_reward is None:
                raise RuntimeError(
                    f"remote verifier returned no score for candidate {index}; "
                    "abort/retry instead of training on a false negative"
                )
            try:
                fraction = float(raw_reward)
            except (TypeError, ValueError):
                fraction = float("nan")
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise RuntimeError(
                    f"remote verifier returned invalid score {raw_reward!r} for candidate "
                    f"{index}; abort/retry instead of poisoning correctness labels"
                )

            correct = math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-12)
            actor_reward = (
                self.binary_reward_weight * float(correct)
                + (1.0 - self.binary_reward_weight) * fraction
                if self.shape_actor_reward
                else float(correct)
            )
            scores.append(
                CandidateScore(
                    correct=correct,
                    actor_reward=actor_reward,
                    details={
                        "task": "code",
                        "provider": self.provider,
                        "fraction_passed": fraction,
                        "pass_all_required": True,
                        "harness_version": STRICT_CODE_HARNESS_VERSION,
                        "binary_reward_weight": self.binary_reward_weight,
                        "actor_reward_shaped": self.shape_actor_reward,
                    },
                )
            )
        return scores


def run_remote_code_sandbox_canary(
    scorer: OpenR1SandboxCodeScorer,
    *,
    include_resource_checks: bool = True,
) -> dict[str, float]:
    """Exercise the configured remote provider before a long training run.

    The canary requires one known-good program to pass and known-bad programs
    to fail for line-boundary changes, missing/extra output, and a nonzero exit.
    By default it also exercises the wall-clock timeout and output cap; that
    adds roughly ``STRICT_CODE_TIMEOUT_SECONDS`` to a real provider call.

    This function deliberately bypasses ``code_reward_fn`` test injection and
    invokes the strict remote provider path. Any missing, malformed, or
    unexpected result raises, so a broken verifier can never look like a batch
    of incorrect model candidates.
    """

    if not isinstance(scorer, OpenR1SandboxCodeScorer):
        raise TypeError("scorer must be an OpenR1SandboxCodeScorer")
    scorer._require_remote_credentials()

    def verification(expected: str) -> dict[str, Any]:
        return {
            "language": "python",
            "test_cases": [
                {
                    "type": "stdin_stdout",
                    "fn_name": None,
                    "input": "ignored\n",
                    "output": expected,
                }
            ],
        }

    names = [
        "known_good",
        "changed_line_boundary",
        "missing_output",
        "extra_output",
        "nonzero_exit",
    ]
    predictions = [
        'print("alpha")\nprint("beta")',
        'print("alpha beta")',
        'print("alpha")',
        'print("alpha")\nprint("beta")\nprint("extra")',
        'print("alpha")\nprint("beta")\nraise SystemExit(9)',
    ]
    infos = [verification("alpha\nbeta\n") for _ in names]
    expected_rewards = [1.0, 0.0, 0.0, 0.0, 0.0]

    if include_resource_checks:
        oversized = "x" * (STRICT_CODE_MAX_OUTPUT_BYTES + 1)
        names.extend(["timeout", "output_cap"])
        predictions.extend(
            [
                "while True:\n    pass",
                (f'import sys\nsys.stdout.write("x" * {STRICT_CODE_MAX_OUTPUT_BYTES + 1})'),
            ]
        )
        infos.extend([verification(""), verification(oversized)])
        expected_rewards.extend([0.0, 0.0])

    try:
        raw_rewards = scorer._strict_remote_rewards(predictions, infos)
    except RewardDependencyError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"remote {scorer.provider} sandbox canary failed before returning results"
        ) from exc
    if raw_rewards is None:
        raise RuntimeError("remote sandbox canary returned no results")
    try:
        values = list(raw_rewards)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("remote sandbox canary returned malformed results") from exc
    if len(values) != len(expected_rewards):
        raise RuntimeError(
            "remote sandbox canary returned the wrong number of results: "
            f"expected {len(expected_rewards)}, got {len(values)}"
        )

    observed: list[float] = []
    for name, value in zip(names, values):
        if value is None:
            raise RuntimeError(f"remote sandbox canary {name} returned no score")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"remote sandbox canary {name} returned malformed score {value!r}"
            ) from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise RuntimeError(f"remote sandbox canary {name} returned invalid score {value!r}")
        observed.append(number)

    mismatches = [
        f"{name}: expected {expected}, got {actual}"
        for name, expected, actual in zip(names, expected_rewards, observed)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
    ]
    if mismatches:
        raise RuntimeError("remote sandbox canary rejected: " + "; ".join(mismatches))
    return dict(zip(names, observed))


@dataclass
class RewardRouter:
    """Runtime bundle for mixed-task reward routing.

    ``code_binary_weight`` is applied after remote scoring, making the actor
    mixture visible in the runner configuration even when a scorer is injected
    for tests.  The default reproduces ``0.8 * binary + 0.2 * fraction``.
    """

    math_scorer: Any
    code_scorer: Optional[Any] = None
    code_binary_weight: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.code_binary_weight <= 1.0:
            raise ValueError("code_binary_weight must lie in [0, 1]")
        self.code_binary_weight = float(self.code_binary_weight)

    def score_math_batch(
        self, predictions: Sequence[object], golds: Sequence[object]
    ) -> list[CandidateScore]:
        return list(self.math_scorer.score_batch(predictions, golds))

    def score_code_batch(
        self,
        predictions: Sequence[str],
        verification_info: Sequence[Mapping[str, Any]],
    ) -> list[CandidateScore]:
        if self.code_scorer is None:
            raise ValueError(
                "code examples require an explicit OpenR1SandboxCodeScorer so the "
                "remote provider choice is visible in the training configuration"
            )
        scores = list(self.code_scorer.score_batch(predictions, verification_info))
        routed: list[CandidateScore] = []
        for score in scores:
            if not isinstance(score, CandidateScore):
                raise TypeError("code scorer must return CandidateScore objects")
            fraction = score.details.get("fraction_passed")
            if fraction is None:
                routed.append(score)
                continue
            fraction = float(fraction)
            actor_reward = (
                self.code_binary_weight * float(score.correct)
                + (1.0 - self.code_binary_weight) * fraction
            )
            details = dict(score.details)
            details["binary_reward_weight"] = self.code_binary_weight
            details["actor_reward_shaped"] = True
            routed.append(CandidateScore(score.correct, actor_reward, details))
        return routed


def _task_name(example: RolloutExample) -> str:
    metadata = example.metadata or {}
    for key in ("task_type", "task", "kind"):
        value = metadata.get(key)
        if value is not None:
            normalized = str(value).lower().strip().replace("-", "_")
            if normalized in _CODE_TASK_NAMES:
                return "code"
            if normalized in _MATH_TASK_NAMES:
                return "math"
            raise ValueError(f"unsupported task type {value!r} for example {example.example_id!r}")
    # The tested Python dataset always carries verification_info.  This is a
    # safer inference than trying to guess from prompt contents.
    return "code" if "verification_info" in metadata else "math"


def _audit_result(score: CandidateScore, prediction: str, reference: str) -> VerificationResult:
    error = score.details.get("error")
    return VerificationResult(
        correct=score.correct,
        predicted=None,
        expected=None,
        predicted_text=str(prediction),
        expected_text=str(reference),
        error=str(error) if error is not None else None,
    )


def rescore_particle_rollout(
    rollout: ParticleRollout,
    examples: Sequence[RolloutExample],
    router: Optional[RewardRouter] = None,
    *,
    response_texts: Optional[Sequence[Sequence[str]]] = None,
    math_scorer: Optional[MathVerifyScorer] = None,
    code_scorer: Optional[OpenR1SandboxCodeScorer] = None,
) -> ParticleRollout:
    """Replace rollout rewards with mixed math/code verifier results.

    Code candidates are flattened into one remote call for throughput.  The
    returned object is the same rollout instance, updated with actor rewards in
    ``rewards``, strict binary labels in ``q_labels``, and
    :class:`VerificationResult` records in ``verification``.
    """

    if router is not None and (math_scorer is not None or code_scorer is not None):
        raise ValueError("provide either router or individual scorers, not both")
    router = router or RewardRouter(
        math_scorer=math_scorer or MathVerifyScorer(),
        code_scorer=code_scorer,
    )

    if len(examples) != rollout.batch_size:
        raise ValueError("examples must match the rollout batch dimension")
    texts = rollout.response_texts if response_texts is None else response_texts
    if len(texts) != rollout.batch_size:
        raise ValueError("response_texts must match the rollout batch dimension")
    normalized_texts = [list(group) for group in texts]
    if any(len(group) != rollout.k for group in normalized_texts):
        raise ValueError("every response_texts group must contain rollout.k candidates")

    math_rows: list[tuple[int, int, str, str]] = []
    code_rows: list[tuple[int, int, str, Mapping[str, Any], str]] = []
    for batch_index, (example, group) in enumerate(zip(examples, normalized_texts)):
        task = _task_name(example)
        if task == "math":
            for branch, prediction in enumerate(group):
                math_rows.append((batch_index, branch, prediction, example.answer))
        else:
            metadata = example.metadata or {}
            info = metadata.get("verification_info")
            if not isinstance(info, Mapping):
                raise ValueError(
                    f"code example {example.example_id!r} is missing mapping verification_info"
                )
            for branch, prediction in enumerate(group):
                code_rows.append((batch_index, branch, prediction, info, example.answer))

    score_grid: list[list[Optional[CandidateScore]]] = [
        [None for _ in range(rollout.k)] for _ in range(rollout.batch_size)
    ]
    if math_rows:
        math_scores = router.score_math_batch(
            [row[2] for row in math_rows], [row[3] for row in math_rows]
        )
        if len(math_scores) != len(math_rows):
            raise RuntimeError("math scorer returned the wrong number of scores")
        for row, score in zip(math_rows, math_scores):
            if not isinstance(score, CandidateScore):
                raise TypeError("math scorer must return CandidateScore objects")
            score_grid[row[0]][row[1]] = score

    if code_rows:
        code_scores = router.score_code_batch(
            [row[2] for row in code_rows], [row[3] for row in code_rows]
        )
        if len(code_scores) != len(code_rows):
            raise RuntimeError("code scorer returned the wrong number of scores")
        for row, score in zip(code_rows, code_scores):
            if not isinstance(score, CandidateScore):
                raise TypeError("code scorer must return CandidateScore objects")
            score_grid[row[0]][row[1]] = score

    completed_grid: list[list[CandidateScore]] = []
    for group in score_grid:
        if any(score is None for score in group):
            raise RuntimeError("internal reward routing error left an unscored candidate")
        completed_grid.append([score for score in group if score is not None])

    reward_values = [[score.actor_reward for score in group] for group in completed_grid]
    label_values = [[float(score.correct) for score in group] for group in completed_grid]
    rollout.rewards = torch.tensor(
        reward_values, device=rollout.rewards.device, dtype=rollout.rewards.dtype
    )
    rollout.q_labels = torch.tensor(
        label_values, device=rollout.rewards.device, dtype=rollout.rewards.dtype
    )
    rollout.response_texts = normalized_texts
    rollout.verification = [
        [
            _audit_result(score, normalized_texts[i][j], examples[i].answer)
            for j, score in enumerate(group)
        ]
        for i, group in enumerate(completed_grid)
    ]
    return rollout


__all__ = [
    "CandidateScore",
    "MathVerifyScorer",
    "OpenR1SandboxCodeScorer",
    "RewardDependencyError",
    "RewardRouter",
    "STRICT_CODE_HARNESS_VERSION",
    "STRICT_CODE_MAX_OUTPUT_BYTES",
    "STRICT_CODE_TIMEOUT_SECONDS",
    "build_strict_stdin_stdout_script",
    "rescore_particle_rollout",
    "run_remote_code_sandbox_canary",
]
