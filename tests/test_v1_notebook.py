from __future__ import annotations

import ast
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "HRM_Text_1B_Particle_V1.ipynb"


def test_v1_notebook_is_valid_and_every_python_cell_compiles() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) >= 20
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"notebook-cell-{index}")
            assert cell.execution_count is None
            assert cell.outputs == []


def test_v1_notebook_contains_every_guarded_stage_and_requested_benchmark() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    for stage in (
        "nccl_probe",
        "shape_smoke",
        "ddp_smoke",
        "vram_plan",
        "noise_scale_sweep",
        "collect_q",
        "q_warmup",
        "rl_pilot",
        '"rl"',
        "final_q_calibration",
        "eval_math",
        "eval_mbpp",
    ):
        assert stage in text
    for benchmark in ("GSM8K", "MATH-500", "GSM-Symbolic", "MBPP+"):
        assert benchmark in text
    assert "GPU_COUNT = 2" in text
    assert "NPROC_ARG = f\"--nproc_per_node={GPU_COUNT}\"" in text
    assert "--nproc_per_node=3" not in text
    assert "v1_2gpu_poc.yaml" in text
    assert '"target_fraction": 0.75' not in text  # read from the measured plan
    assert "measured_peak_fraction_max" in text
    assert "q_ready=False" in text
    assert "Q never supplies actor reward" in text
    assert "RUN_EVALPLUS_DOCKER = False" in text
    assert "v1_code_provider_probe.py" in text
    assert "v1_token_audit.py" in text
    assert "v1_summarize_evalplus.py" in text
    assert "open-r1[code]" in text


def test_notebook_never_executes_generated_python_on_the_host() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "exec(" not in code
    assert "eval(" not in code
    assert 'mbpp_generation["docker_commands"]' in code
    runner = (ROOT / "src" / "hrm_particle" / "v1_distributed.py").read_text()
    assert "ganler/evalplus:latest" not in runner
    assert "ganler/evalplus@sha256:" in runner
