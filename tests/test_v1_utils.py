from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from hrm_particle.v1_utils import (
    BF16MasterAdamW,
    atomic_torch_save,
    calibrate_q_margin,
    code_signal_gate,
    deterministic_math_majority_vote,
    deterministic_prompt_split,
    extract_python_code,
    prompt_split_masks,
    stratified_prompt_split_masks,
    stratified_prompt_splits,
    select_with_anchor_fallback,
    sha256_file,
)


def test_q_prompt_splits_are_deterministic_disjoint_and_near_requested_sizes() -> None:
    percentages = {
        "train": 70,
        "early_stop": 10,
        "margin_select": 10,
        "safety_test": 10,
    }
    ids = [f"source:prompt-{index}" for index in range(10_000)]
    first = prompt_split_masks(ids, percentages)
    second = prompt_split_masks(ids, percentages)
    assert all(torch.equal(first[name], second[name]) for name in percentages)
    assert torch.stack(list(first.values())).sum(dim=0).eq(1).all()
    assert 6_800 <= int(first["train"].sum()) <= 7_200
    for name in ("early_stop", "margin_select", "safety_test"):
        assert 850 <= int(first[name].sum()) <= 1_150
    assert deterministic_prompt_split(ids[7], percentages) in percentages


@pytest.mark.parametrize(
    "percentages",
    [
        {"train": 70, "early_stop": 10, "margin_select": 10},
        {"train": 70, "early_stop": 10, "margin_select": 10, "safety_test": 9},
        {"train": 70, "early_stop": 10, "margin_select": 10, "safety_test": True},
    ],
)
def test_q_prompt_split_rejects_malformed_percentages(percentages: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        deterministic_prompt_split("id", percentages)


def test_stratified_q_split_has_exact_source_balanced_heldout_counts() -> None:
    percentages = {
        "train": 70,
        "early_stop": 10,
        "margin_select": 10,
        "safety_test": 10,
    }
    ids = [f"math:{index}" for index in range(512)] + [
        f"code:{index}" for index in range(512)
    ]
    strata = ["math"] * 512 + ["code"] * 512
    first = stratified_prompt_splits(ids, strata, percentages)
    second = stratified_prompt_splits(ids, strata, percentages)
    assert first == second
    masks = stratified_prompt_split_masks(ids, strata, percentages)
    for stratum_start in (0, 512):
        for name in ("early_stop", "margin_select", "safety_test"):
            assert int(masks[name][stratum_start : stratum_start + 512].sum()) == 51
    assert torch.stack(list(masks.values())).sum(dim=0).eq(1).all()


def test_extract_python_code_prefers_longest_python_fence() -> None:
    response = """
An illustration:
```python
x = 1
```

Final answer:
```python linenums
def add(a, b):
    return a + b
```

```javascript
throw new Error("not Python")
```
"""
    assert extract_python_code(response) == "def add(a, b):\n    return a + b"


def test_extract_python_code_unlabelled_raw_and_foreign_only() -> None:
    assert extract_python_code("~~~\ndef f():\n    return 3\n~~~") == (
        "def f():\n    return 3"
    )
    assert extract_python_code("  def f():\n    return 3  ") == "def f():\n    return 3"
    assert extract_python_code("```javascript\nconst x = 1;\n```") == ""
    with pytest.raises(TypeError, match="string"):
        extract_python_code(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["2", "3", "3", "2"], 0),  # tied counts; earliest sample wins
        ([None, "3", "3", "2"], 1),
        ([None, "", None, ""], 0),
        (["2", "3", "2", "4"], 0),
    ],
)
def test_deterministic_majority_vote(
    answers: list[str | None], expected: int
) -> None:
    assert deterministic_math_majority_vote(answers) == expected


def test_deterministic_majority_vote_validates_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        deterministic_math_majority_vote([])
    with pytest.raises(TypeError, match="sequence"):
        deterministic_math_majority_vote("2")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string or None"):
        deterministic_math_majority_vote([2])  # type: ignore[list-item]


def test_anchor_fallback_requires_readiness_and_strict_margin() -> None:
    logits = torch.tensor(
        [
            [0.0, 1.0, 0.5],
            [1.0, 1.5, 1.4],  # equality at a 0.5 margin falls back
            [0.0, 0.2, 0.3],
        ]
    )
    assert select_with_anchor_fallback(logits, 0.5, False).tolist() == [0, 0, 0]
    assert select_with_anchor_fallback(logits, 0.5, True).tolist() == [1, 0, 0]
    ready = torch.tensor([False, True, True])
    assert select_with_anchor_fallback(logits, 0.1, ready).tolist() == [0, 1, 2]


def test_anchor_fallback_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        select_with_anchor_fallback(torch.ones(4), 0.0, True)
    with pytest.raises(ValueError, match="finite"):
        select_with_anchor_fallback(torch.tensor([[0.0, float("nan")]]), 0.0, True)
    with pytest.raises(ValueError, match="non-negative"):
        select_with_anchor_fallback(torch.zeros(1, 2), -0.1, True)
    with pytest.raises(TypeError, match="dtype"):
        select_with_anchor_fallback(torch.zeros(1, 2), 0.0, torch.ones(1))


def _safe_calibration_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    # On the first forty prompts the explorer rescues an incorrect anchor and
    # has a large Q gap.  On the remaining prompts Q correctly retains branch 0.
    logits = torch.zeros(100, 2)
    labels = torch.zeros(100, 2)
    logits[:40, 1] = 1.0
    labels[:40, 1] = 1.0
    logits[40:, 1] = -1.0
    labels[40:, 0] = 1.0
    labels[40:, 1] = 0.0
    return logits, labels


def test_margin_calibration_is_reproducible_and_passes_safe_selector() -> None:
    logits, labels = _safe_calibration_fixture()
    kwargs = dict(
        margins=(0.0, 0.5, 1.0),
        bootstrap_samples=500,
        seed=17,
        min_prompts=64,
        min_switches=10,
    )
    first = calibrate_q_margin(logits, labels, **kwargs)
    second = calibrate_q_margin(logits, labels, **kwargs)
    assert first == second
    assert first.ready
    assert first.margin == 0.5  # equal accuracy; larger safe margin wins
    assert first.anchor_accuracy == pytest.approx(0.6)
    assert first.selected_accuracy == pytest.approx(1.0)
    assert first.mean_delta == pytest.approx(0.4)
    assert first.ci_low > 0.0
    assert first.switch_count == 40
    assert len(first.trials) == 3


def test_margin_calibration_blocks_harmful_or_underpowered_q() -> None:
    harmful_logits = torch.tensor([[0.0, 1.0]]).repeat(100, 1)
    harmful_labels = torch.tensor([[1.0, 0.0]]).repeat(100, 1)
    harmful = calibrate_q_margin(
        harmful_logits,
        harmful_labels,
        margins=(0.0,),
        bootstrap_samples=200,
        min_prompts=64,
    )
    assert not harmful.ready
    assert harmful.ci_high < 0
    assert "no margin passed" in harmful.reason

    logits, labels = _safe_calibration_fixture()
    underpowered = calibrate_q_margin(
        logits[:8],
        labels[:8],
        margins=(0.0,),
        bootstrap_samples=200,
        min_prompts=64,
    )
    assert not underpowered.ready
    assert "64" in underpowered.reason
    # The consumer-side safety invariant is independent of diagnostic scores.
    assert select_with_anchor_fallback(logits[:8], underpowered.margin, underpowered.ready).eq(0).all()


def test_margin_calibration_validates_binary_labels() -> None:
    logits = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="binary"):
        calibrate_q_margin(
            logits,
            torch.full_like(logits, 0.5),
            bootstrap_samples=100,
            min_prompts=1,
        )
    with pytest.raises(ValueError, match="same shape"):
        calibrate_q_margin(
            logits,
            torch.zeros(3, 2),
            bootstrap_samples=100,
            min_prompts=1,
        )


def test_code_signal_gate_enables_only_with_nonzero_and_mixed_groups() -> None:
    labels = torch.zeros(40, 4)
    labels[:8, 1] = 1.0
    passed = code_signal_gate(labels)
    assert passed.enabled
    assert passed.nonzero_groups == 8
    assert passed.mixed_groups == 8
    assert passed.nonzero_fraction == pytest.approx(0.2)

    failed = code_signal_gate(torch.zeros(40, 4))
    assert not failed.enabled
    assert "nonzero groups" in failed.reason
    assert "mixed groups" in failed.reason


def test_code_signal_gate_rejects_fractional_correctness() -> None:
    with pytest.raises(ValueError, match="binary"):
        code_signal_gate(torch.tensor([[0.0, 0.2, 1.0, 0.0]]))


def _assert_master_state_is_fp32(optimizer: BF16MasterAdamW) -> None:
    assert all(parameter.dtype == torch.float32 for parameter in optimizer.master_parameters)
    for state in optimizer.master_optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                assert value.dtype == torch.float32


def test_bf16_master_adamw_updates_and_round_trips_exactly() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    optimizer = BF16MasterAdamW([parameter], lr=0.1, weight_decay=0.0)
    assert optimizer.param_groups[0]["params"][0] is parameter

    parameter.grad = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    optimizer.step()
    assert torch.equal(parameter, optimizer.master_parameters[0].to(torch.bfloat16))
    _assert_master_state_is_fp32(optimizer)

    state = optimizer.state_dict()
    restored_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    restored = BF16MasterAdamW([restored_parameter], lr=9.0, weight_decay=1.0)
    restored.load_state_dict(state)
    assert torch.equal(restored_parameter, parameter)
    assert restored.param_groups[0]["lr"] == pytest.approx(0.1)
    _assert_master_state_is_fp32(restored)

    next_gradient = torch.tensor([-0.75, 0.125], dtype=torch.bfloat16)
    parameter.grad = next_gradient.clone()
    restored_parameter.grad = next_gradient.clone()
    optimizer.step()
    restored.step()
    assert torch.equal(restored_parameter, parameter)
    assert torch.equal(restored.master_parameters[0], optimizer.master_parameters[0])


def test_bf16_master_adamw_honors_public_scheduler_lr_and_zero_grad() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = BF16MasterAdamW([parameter], lr=0.1, weight_decay=0.0)
    optimizer.param_groups[0]["lr"] = 0.025
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert optimizer.master_optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
    optimizer.zero_grad(set_to_none=True)
    assert parameter.grad is None
    assert optimizer.master_parameters[0].grad is None


def test_bf16_master_adamw_rejects_non_bf16_and_bad_resume_shape() -> None:
    with pytest.raises(TypeError, match="bfloat16"):
        BF16MasterAdamW([torch.nn.Parameter(torch.ones(2))])

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = BF16MasterAdamW([parameter])
    state = optimizer.state_dict()
    state["master_weights"] = [torch.ones(3)]
    with pytest.raises(ValueError, match="shape"):
        optimizer.load_state_dict(state)
    with pytest.raises(RuntimeError, match="fixed"):
        optimizer.add_param_group(
            {"params": [torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))]}
        )


def test_atomic_torch_save_and_sha256(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "checkpoint.pt"
    payload = {"step": 7, "tensor": torch.arange(5)}
    assert atomic_torch_save(payload, destination) == destination
    loaded = torch.load(destination, weights_only=False)
    assert loaded["step"] == 7
    assert torch.equal(loaded["tensor"], payload["tensor"])

    expected = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert sha256_file(destination, chunk_size=3) == expected
    assert len(expected) == 64


def test_failed_atomic_save_preserves_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"known-good")

    def fail_after_partial_write(payload: object, handle: object) -> None:
        handle.write(b"partial")  # type: ignore[attr-defined]
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated"):
        atomic_torch_save({"step": 8}, destination)
    assert destination.read_bytes() == b"known-good"
    assert list(tmp_path.glob(".checkpoint.pt.tmp-*")) == []


def test_sha256_validates_path_and_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing")
    source = tmp_path / "payload"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="positive"):
        sha256_file(source, chunk_size=0)
