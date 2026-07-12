from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hrm_particle.data import generate_synthetic_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("KMP_USE_SHM", "0")
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _dataset(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    generate_synthetic_dataset(
        data,
        sizes={"train": 12, "dev": 6, "test": 6, "ood": 6},
        train_seed="train",
        eval_seed="eval",
        min_difficulty=1,
        max_difficulty=2,
    )
    return data


def test_prepare_data_cli_needs_private_eval_seed(tmp_path):
    output = tmp_path / "missing-seed"
    result = _run(
        "scripts/prepare_data.py",
        "synthetic",
        "--output-dir",
        str(output),
        "--train-count",
        "6",
        "--dev-count",
        "6",
        "--test-count",
        "6",
        "--ood-count",
        "6",
    )
    assert result.returncode == 2
    assert "private evaluation seed is required" in result.stderr
    assert not output.exists()


def test_prepare_data_cli_and_validate(tmp_path):
    seed = tmp_path / "eval-seed.txt"
    seed.write_text("private-test-seed\n", encoding="utf-8")
    seed.chmod(0o600)
    output = tmp_path / "prepared"
    generated = _run(
        "scripts/prepare_data.py",
        "synthetic",
        "--output-dir",
        str(output),
        "--eval-seed-file",
        str(seed),
        "--train-count",
        "12",
        "--dev-count",
        "6",
        "--test-count",
        "6",
        "--ood-count",
        "6",
    )
    assert generated.returncode == 0, generated.stderr
    assert "private-test-seed" not in generated.stdout
    status = (output / "DATA_STATUS.txt").read_text(encoding="utf-8")
    assert "PRIVATE USER-SUPPLIED" in status
    assert "private-test-seed" not in status
    validated = _run("scripts/prepare_data.py", "validate", "--data-dir", str(output))
    assert validated.returncode == 0, validated.stderr
    payload = json.loads(validated.stdout)
    assert payload["isolation"]["records"] == 30


@pytest.mark.parametrize("config", ["configs/poc_h100.yaml", "configs/poc_a100.yaml"])
def test_train_launcher_dry_run_validates_data_and_budget(tmp_path, config):
    data = _dataset(tmp_path)
    result = _run(
        "scripts/train.py",
        "--config",
        config,
        "--set",
        f"data.directory={data}",
        "--output-dir",
        str(tmp_path / "run"),
        "--max-cost-usd",
        "5",
        "--max-gpu-hours",
        "1",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["data"]["records"] == 30
    assert payload["budget"]["projected_cost_usd"] <= 5
    assert payload["evaluation_seed_status"] == "private_user_supplied"


def test_train_launcher_refuses_budget_increase(tmp_path):
    data = _dataset(tmp_path)
    result = _run(
        "scripts/train.py",
        "--config",
        "configs/poc_h100.yaml",
        "--set",
        f"data.directory={data}",
        "--max-cost-usd",
        "41",
        "--dry-run",
    )
    assert result.returncode == 2
    assert "may only lower" in result.stderr

    generic_override = _run(
        "scripts/train.py",
        "--config",
        "configs/poc_h100.yaml",
        "--set",
        f"data.directory={data}",
        "--set",
        "budget.hourly_usd=0.01",
        "--dry-run",
    )
    assert generic_override.returncode == 2
    assert "budget.* cannot be changed" in generic_override.stderr

    nucleus_training = _run(
        "scripts/train.py",
        "--config",
        "configs/poc_h100.yaml",
        "--set",
        f"data.directory={data}",
        "--set",
        "generation.top_p=0.95",
        "--dry-run",
    )
    assert nucleus_training.returncode == 2
    assert "finite support" in nucleus_training.stderr


def test_evaluate_launcher_dry_run(tmp_path):
    data = _dataset(tmp_path)
    result = _run(
        "scripts/evaluate.py",
        "--config",
        "configs/eval.yaml",
        "--set",
        f"data.directory={data}",
        "--output-dir",
        str(tmp_path / "eval"),
        "--split",
        "ood",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["splits"] == ["ood"]
    assert payload["data"]["records"] == 30
    assert payload["evaluation_seed_status"] == "private_user_supplied"


@pytest.mark.integration
def test_offline_smoke_script_runs_optimizer_steps():
    scripts_path = str(PROJECT_ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from run_smoke import _data_smoke, _tiny_training_smoke

    data = _data_smoke()
    tiny_training = _tiny_training_smoke()
    assert data["records"] == 60
    assert tiny_training["anchor_bit_exact"] is True
    assert tiny_training["q_ranking_pairs"] > 0
