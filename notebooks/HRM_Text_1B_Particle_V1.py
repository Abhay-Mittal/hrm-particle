# %% [markdown]
# # HRM-Text-1B: direct Gaussian H-particles + correctness Q, V1
#
# This notebook is the **control plane** for a restartable two-GPU POC.
# It is accelerator-model agnostic: H100/H200, GH200, B200, or another CUDA
# GPU is acceptable if it supports BF16/NCCL and has enough memory.
# Long jobs run in fresh `torchrun` subprocesses; the notebook kernel never owns
# the 1B model or a distributed process group.
#
# The experiment tests a narrow hypothesis:
#
# 1. keep branch 0 as the exact pretrained HRM policy (`epsilon=0`, greedy decode);
# 2. create three coherent response-level branches by adding one normalized,
#    hidden-sized Gaussian direction after H1 and reusing it for the whole answer;
# 3. split a disjoint development slice into scale-selection and confirmation
#    halves, choose on selection only, and require confirmation no-degradation;
#    the default Gaussian particle has no learned actor parameters;
# 4. train Q separately as **binary correctness prediction** on detached final
#    H states; Q never supplies actor reward;
# 5. let Q leave the anchor only after a held-out no-degradation gate.
#
# This is the deliberately simpler PTRM-style V1. Set `particles.mode: learned`
# only for the optional query-conditioned adapter + verifier-RL extension.
#
# A random Q head therefore cannot lower the initial policy: it is a sidecar,
# its last layer starts at exactly zero, ties select branch 0, and `q_ready=False`
# forces branch 0 until calibration passes.

# %% [markdown]
# ## What is being trained?
#
# | object | shape / size | update rule |
# |---|---:|---|
# | HRM-Text-1B | 1,182,795,264 parameters; hidden 1,536 | frozen |
# | H and L stacks | 16 physical layers each; H=2, L=3 recurrence | frozen |
# | Gaussian H-particle | `[B,4,1536]`; branch 0 exactly zero | sampled once per answer; fixed |
# | noise scale | candidates 0--0.05 relative H-state RMS | task-balanced select/confirm halves |
# | learned adapter (optional) | query/latent bottleneck 64; RMS bounded at 0.10 | verifier RL |
# | Q head | 1,536→256 twice→1; 786,945 parameters | correctness BCE + ranking |
#
# The intervention is after the first call to `H_module`, before the remaining
# L/H computation. Prompt tokens are never perturbed. A fixed causal
# `"\nSolution:\n"` prefix is the first injected position, so the particle can
# change the first sampled content token without changing the bidirectional
# PrefixLM prompt.
#
# **BF16 policy:** backbone, Q weights, and model activations are BF16. Gaussian
# normalization and scale arithmetic are FP32 before the bounded delta is cast
# to the H-state dtype. BCE/reductions, verifier arithmetic, and AdamW master
# weights/state remain FP32. The optional learned mode also keeps PPO/KL math
# and adapter optimizer state in FP32.

# %%
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import yaml

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    raise RuntimeError("Start this notebook from the hrm-particle-poc project root.")

GPU_COUNT = 2  # tested values: 2 (POC default) or 3
if GPU_COUNT not in {2, 3}:
    raise ValueError("Use the tested two- or three-GPU configuration.")
profile = {
    2: {
        "config": "v1_2gpu_poc.yaml",
        "run": "v1_2gpu_poc",
        "data": "v1_poc",
        "preset": "poc",
        "rl_hours": 8,
        "rl_hours_range": (6, 10),
    },
    3: {
        "config": "v1_3gpu_full.yaml",
        "run": "v1_3gpu_full",
        "data": "v1",
        "preset": "full",
        "rl_hours": 12,
        "rl_hours_range": (12, 20),
    },
}[GPU_COUNT]
CONFIG = PROJECT_ROOT / "configs" / profile["config"]
PYTHON = sys.executable
RUN_DIR = PROJECT_ROOT / "runs" / profile["run"]
DATA_DIR = PROJECT_ROOT / "data" / profile["data"]
DATA_PRESET = profile["preset"]
NPROC_ARG = f"--nproc_per_node={GPU_COUNT}"

# Do not import torch or initialize CUDA in this notebook. Every GPU stage is a
# fresh process, which avoids notebook CUDA state leaking into torchrun.
print({
    "project": str(PROJECT_ROOT),
    "python": PYTHON,
    "config": str(CONFIG),
    "gpu_count": GPU_COUNT,
    "gpu_type": "not hard-coded; CUDA + BF16 + NCCL required",
})

# %% [markdown]
# ## 1. Install and restart the kernel
#
# Run this once in the target CUDA environment, then restart the notebook kernel.
# Transformers must expose native `HrmTextForCausalLM`; the checkpoint has no
# remote-code fallback. For code RL, install Open-R1 from a **reviewed commit**
# and configure E2B or Morph. The commit is recorded by `pip freeze` in the run
# provenance; do not silently use a moving branch for a reported result.

# %%
INSTALL = False  # set True for a fresh environment, run once, then restart kernel
OPEN_R1_COMMIT = os.environ.get(
    "OPEN_R1_COMMIT", "1416fa0cf21595d2083b399a2a0bbddd7f6e9563"
)  # reviewed full git SHA; must match the YAML

if INSTALL:
    subprocess.run(
        [PYTHON, "-m", "pip", "install", "--upgrade", "pip", "wheel"], check=True
    )
    subprocess.run(
        [PYTHON, "-m", "pip", "install", "-r", "requirements-v1.txt"], check=True
    )
    subprocess.run([PYTHON, "-m", "pip", "install", "-e", "."], check=True)
    if len(OPEN_R1_COMMIT) != 40 or any(
        character not in "0123456789abcdef" for character in OPEN_R1_COMMIT
    ):
        raise ValueError("OPEN_R1_COMMIT must be one reviewed full lowercase Git SHA.")
    subprocess.run(
        [
            PYTHON,
            "-m",
            "pip",
            "install",
            f"open-r1[code] @ git+https://github.com/huggingface/open-r1.git@{OPEN_R1_COMMIT}",
        ],
        check=True,
    )
    print("Restart the kernel now before continuing.")
else:
    print("Install skipped. Set INSTALL=True only in a fresh environment.")

# %% [markdown]
# ## 2. CPU-only configuration and dependency check
#
# `validate` does not load a model or touch CUDA. The default POC preset uses:
#
# - direct, response-fixed Gaussian H-state particles with no trainable adapter;
# - 128 unique `rl_train` scale prompts split deterministically into disjoint
#   selection/confirmation halves within math and code, interleaved by source;
# - Q data: 2,048 groups × K=4 = 8,192 states (5,700+ train candidates);
# - RL pool: 2,400 verified math + 600 verified Python prompts;
# - quick evaluation: 500 GSM8K, 200 MATH-500, 200 GSM-Symbolic, plus MBPP+;
# - an optional learned-mode 8-hour actor-RL cap. Gaussian mode skips actor RL
#   and spends the budget on scale selection, Q warmup, and matched evaluation.

# %%
def run(command, *, env=None, cwd=PROJECT_ROOT):
    print("$", " ".join(map(str, command)), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    return subprocess.run(list(map(str, command)), cwd=cwd, env=merged, check=True)


run([PYTHON, "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "validate"])
run([PYTHON, "scripts/v1_prepare_data.py", "show-preset", "--preset", DATA_PRESET])

CONFIG_DATA = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
PARTICLE_MODE = str(CONFIG_DATA["particles"]["mode"])
if PARTICLE_MODE not in {"gaussian", "learned"}:
    raise ValueError(f"Unsupported particles.mode: {PARTICLE_MODE!r}")
Q_ONLY_CHECKPOINT = (
    PROJECT_ROOT / CONFIG_DATA["paths"]["q_checkpoint"]
    if PARTICLE_MODE == "gaussian"
    else None
)


def inference_checkpoint_args():
    """Use the standalone Q artifact for fixed-Gaussian inference."""

    return ["--checkpoint", Q_ONLY_CHECKPOINT] if Q_ONLY_CHECKPOINT else []


print({
    "particle_mode": PARTICLE_MODE,
    "inference_checkpoint": (
        str(Q_ONLY_CHECKPOINT) if Q_ONLY_CHECKPOINT else "trained actor checkpoint"
    ),
})

# %% [markdown]
# ## 3. Prepare pinned and decontamination-audited data
#
# Preparation resolves the reviewed dataset revisions, normalizes schemas,
# deduplicates, removes 13-token overlap against every evaluation prompt, and
# writes SHA256 manifests. No generated or reference code is executed.
#
# Training sources:
#
# - math: DAPO-Math-17k English, with easier GSM8K/MATH items in Q warmup;
# - code: Open-R1's tested, Python-only, decontaminated verifiable-code set.
#
# MBPP+ is evaluation-only and included in the decontamination index. Public
# benchmarks are described as **decontamination-audited**, not guaranteed
# uncontaminated: the released HRM corpus snapshot post-dates these benchmarks.
# Dataset cards/manifest metadata are retained for review. In particular, the
# processed DAPO and verified-code derivatives do not provide one simple license
# guarantee for every upstream row; review upstream terms before commercial use.

# %%
PREPARE_DATA = False  # networked; set True once
if PREPARE_DATA:
    run(
        [
            PYTHON,
            "scripts/v1_prepare_data.py",
            "prepare",
            "--output-dir",
            DATA_DIR,
            "--preset",
            DATA_PRESET,
            "--seed",
            "20260710-v1",
            "--include-mbpp-plus-contamination",
        ]
    )

run([PYTHON, "scripts/v1_prepare_data.py", "validate", "--data-dir", DATA_DIR])
run([
    PYTHON, "scripts/v1_token_audit.py", "--config", CONFIG,
    "--output", RUN_DIR / "token-audit.json",
])
manifest = json.loads((DATA_DIR / "manifest.json").read_text())
print(json.dumps({
    "q_warm": manifest["q_warm"],
    "planned_counts": manifest["planned_counts"],
    "decontamination": manifest["decontamination"],
    "safety": manifest["safety"],
}, indent=2))

# %% [markdown]
# ## 4. Offline test gate
#
# These tests cover direct Gaussian dimensions/normalization, exact-zero
# injection, H-hook placement, cache isolation, replay ratios,
# correctness/actor target separation, Q cold-start selection,
# FP32-master BF16 AdamW, data schemas/decontamination, sandbox fail-closed
# behavior, dynamic rank-batch disjointness, exact `[batch,K=4]` reward-slot
# alignment, proof that Q logits/labels cannot alter actor loss, final-state Q
# gating, checkpoint round trips, and the real tiny HRM integration when the
# installed Transformers build contains HRM.

# %%
run([PYTHON, "-m", "pytest", "-q"])
run([
    PYTHON, "-m", "ruff", "check",
    "src/hrm_particle", "scripts/v1_distributed.py", "scripts/v1_prepare_data.py",
    "scripts/v1_code_provider_probe.py", "scripts/v1_summarize_evalplus.py",
    "scripts/v1_token_audit.py",
])

# %% [markdown]
# ## 5. Hardware and real-model smoke tests
#
# The first command checks NCCL across the configured two or three ranks. The second loads the
# official 1B checkpoint on one GPU and passes dummy text through the exact
# PrefixLM/H1-injection path. It asserts:
#
# - official parameter count/config/dimensions;
# - zero-particle logits are bit-identical to the clean model;
# - prompt deltas are exactly zero and explorer RMS is bounded;
# - terminal/query shapes are `[1,4,1536]`, Q is `[1,4]`;
# - Gaussian mode has no adapter parameters and Q receives finite gradients
#   (learned mode additionally checks adapter gradients and replay ratio one).
#
# The third command performs one synchronized update on all configured GPUs and
# checks the trainable/Q checksum is identical on every rank. The final command measures
# a real longest-prompt rollout, reference replay, and backward pass on every
# rank. It chooses the largest shared prompt batch near 75% VRAM, never above
# the 80% hard cap, and adjusts accumulation to preserve the effective batch.

# %%
GPU_ENV = {
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
run(["nvidia-smi", "-L"])
run([
    "torchrun", "--standalone", NPROC_ARG,
    "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "nccl_probe",
], env=GPU_ENV)

# %%
run([
    PYTHON, "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "shape_smoke",
], env={**GPU_ENV, "CUDA_VISIBLE_DEVICES": "0"})

# %%
run([
    "torchrun", "--standalone", NPROC_ARG,
    "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "ddp_smoke",
], env=GPU_ENV)

# %%
REPLAN_MEMORY = False  # True only when hardware changed and no Q/RL artifacts exist
memory_command = [
    "torchrun", "--standalone", NPROC_ARG,
    "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "vram_plan",
]
if REPLAN_MEMORY:
    memory_command.append("--replan-memory")
run(memory_command, env=GPU_ENV)
memory_plan = json.loads((RUN_DIR / "memory-plan.json").read_text())
print(json.dumps({
    "gpu_model_agnostic": True,
    "rl_prompt_micro_batch_size": memory_plan["rl_prompt_micro_batch_size"],
    "gradient_accumulation_steps": memory_plan["rl_gradient_accumulation_steps"],
    "effective_global_prompt_groups": memory_plan[
        "effective_global_prompt_groups_per_update"
    ],
    "measured_peak_fraction_max": memory_plan["measured_peak_fraction_max"],
    "measured_minimum_free_gib": memory_plan["measured_minimum_free_gib"],
}, indent=2))

# %% [markdown]
# ## 6. Select Gaussian scale, then collect 8,192 correctness-labelled Q examples
#
# In the default mode, a predeclared scale sweep first compares
# `0, .005, .01, .02, .03, .05` on 128 unique prompts drawn from `rl_train`.
# The runner builds disjoint, deterministic 50/50 selection and confirmation
# halves separately within math and code, using deterministic source interleaving
# so one source cannot fill a half in a block. Every scale gets the same prompt
# IDs and random directions. Scale zero is the matched control; final benchmarks
# and Q calibration data are never used for tuning.
#
# The scale is chosen **only on the selection half**: take the smallest positive
# scale within `oracle_tie_tolerance` of the best strict-correctness oracle@4.
# If every positive scale underperforms scale zero there, the stage aborts instead
# of forcing harmful noise. The chosen scale then faces a no-degradation gate on
# the untouched confirmation half: its point-estimate oracle delta versus zero
# must be nonnegative overall **and separately for math and code**, or the run
# aborts. `development_evidence_positive` is computed only there and is
# the stronger condition that the paired-bootstrap lower bound is above zero;
# it is recorded, not required for this POC. Confirmation never reselects the
# scale. This is auditable model selection, not actor training.
#
# Then each of 2,048 prompt groups (512 each from GSM8K, MATH, DAPO, and verified
# Python) produces K=4 on-policy candidates. Math is scored by Math-Verify.
# Python is sent in one batch per rollout to Open-R1's E2B/Morph provider using
# an exact-count stdin/stdout harness; there is no local fallback. A known-good
# sentinel accompanies every remote batch so provider outages abort. The saved BF16 tensors are detached
# `(prompt_summary, terminal_state)` pairs plus strict binary correctness.
#
# Q warmup is supervised outcome classification—not language-model SFT. It uses
# BCE on every candidate plus a 0.1 correct-vs-wrong within-prompt ranking loss.

# %%
provider = "e2b"  # must match the selected CONFIG
required_key = "E2B_API_KEY" if provider == "e2b" else "MORPH_API_KEY"
if not os.environ.get(required_key):
    raise RuntimeError(
        f"Set {required_key}. Code candidates are never executed on the training host."
    )
run([
    PYTHON, "scripts/v1_code_provider_probe.py", "--config", CONFIG,
    "--output", RUN_DIR / "code-provider-canary.json",
])

if PARTICLE_MODE == "gaussian":
    run([
        "torchrun", "--standalone", NPROC_ARG,
        "scripts/v1_distributed.py", "--config", CONFIG,
        "--stage", "noise_scale_sweep",
    ], env=GPU_ENV)
    noise_scale_selection = json.loads(
        (RUN_DIR / "noise-scale-selection.json").read_text(encoding="utf-8")
    )
    print(json.dumps(noise_scale_selection, indent=2))
else:
    print("Learned mode: fixed-Gaussian scale selection is not applicable.")

run([
    "torchrun", "--standalone", NPROC_ARG,
    "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "collect_q",
], env=GPU_ENV)

# %% [markdown]
# ## 7. Warm Q on one GPU and calibrate anchor fallback
#
# Q's final weight is exactly zero and its bias starts at the empirical success
# prior. Inner projections may be Xavier because they cannot affect output until
# the last layer moves. Source-stratified prompt groups—not candidates—are split
# 70/10/10/10 for training, epoch selection, margin choice, and an untouched
# safety test. All K branches stay together. A paired bootstrap certifies the
# already-chosen margin on the last split; otherwise `q_ready=False`.

# %%
run([
    PYTHON, "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "q_warmup",
], env={**GPU_ENV, "CUDA_VISIBLE_DEVICES": "0"})
q_calibration = json.loads((RUN_DIR / "q-calibration.json").read_text())
print(json.dumps({
    "q_ready": q_calibration["q_ready"],
    "margin": q_calibration["margin"],
    "anchor_accuracy": q_calibration["anchor_accuracy"],
    "selected_accuracy": q_calibration["selected_accuracy"],
    "ci": [q_calibration["ci_low"], q_calibration["ci_high"]],
    "positive_rate_train": q_calibration["positive_rate_train"],
    "split_prompt_groups": q_calibration["split_prompt_groups"],
}, indent=2))

# %% [markdown]
# ## 8. Optional learned-adapter RL/code-signal pilot
#
# **Default Gaussian mode skips this section.** Its H-particle and selected scale
# are fixed, so there are no actor parameters to update and running PPO would be
# scientifically misleading. Q was already trained in the previous section.
#
# In optional `particles.mode: learned`, HRM-Text-1B's model card says it was not
# trained on code, so the pilot measures strict pass-all groups before committing
# the full run. If too few code groups have any/mixed successes, the full stage
# automatically disables code RL and continues with math. It does not pretend
# all-zero code groups contain a policy-gradient signal.
# The same pilot records end-to-end update times (including remote verification).
# A max-time × 1.25 forecast must leave room for at least 40 full-RL updates;
# otherwise the notebook stops before spending the long-run budget.
#
# Actor reward for code is `0.8*all_tests_pass + 0.2*fraction_tests_passed`.
# **Q label remains `all_tests_pass` only.** Math actor/Q rewards are both strict
# correctness. Q is updated online on detached rollout states; it is still never
# part of the PPO advantage.

# %%
if PARTICLE_MODE == "learned":
    run([
        "torchrun", "--standalone", NPROC_ARG,
        "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "rl_pilot",
    ], env=GPU_ENV)
    code_gate = json.loads((RUN_DIR / "pilot" / "code-signal.json").read_text())
    print(json.dumps(code_gate, indent=2))

    pilot_rows = [
        json.loads(line)
        for line in (RUN_DIR / "pilot" / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    pilot_update_seconds = [float(row["update_seconds"]) for row in pilot_rows]
    if not pilot_update_seconds:
        raise RuntimeError("Pilot produced no update timings.")
    conservative_update_seconds = max(pilot_update_seconds) * 1.25
    stop_buffer_minutes = float(CONFIG_DATA["runtime"]["stop_buffer_minutes"])
    forecast_updates = int(
        (profile["rl_hours"] * 3600 - stop_buffer_minutes * 60)
        / conservative_update_seconds
    )
    MINIMUM_FORECAST_RL_UPDATES = 40
    print({
        "pilot_updates": len(pilot_update_seconds),
        "maximum_pilot_update_seconds": max(pilot_update_seconds),
        "conservative_seconds_per_update": conservative_update_seconds,
        "forecast_updates_in_budget": forecast_updates,
    })
    if forecast_updates < MINIMUM_FORECAST_RL_UPDATES:
        raise RuntimeError(
            "Pilot throughput forecasts too few full-RL updates for a useful V1. "
            "Shorten generation or investigate provider/model throughput first."
        )
else:
    print("Gaussian mode: skipping actor pilot; the H-particle is parameter-free.")

# %% [markdown]
# ## 9. Optional time-boxed learned-adapter RL run (restartable)
#
# **Default Gaussian mode skips this section** and proceeds directly to fresh Q
# calibration and evaluation using the standalone Q checkpoint.
#
# Learned mode's two-GPU POC defaults to an 8-hour cumulative full-RL cap. Together with
# setup, Q warmup, calibration, and quick evaluation, the target is roughly
# 12–15 hours end-to-end on recent high-end BF16 GPUs. Hardware and provider
# throughput vary, so this is an estimate rather than a guaranteed deadline.
# Rank 0 writes an
# atomic adapter/Q-only checkpoint every 20 updates, including both FP32-master
# optimizer states, exact data cursor (`step`), hashes, and per-rank Python/CPU/
# CUDA/token/latent RNG state. Resume rejects world-size, model revision, data,
# Q calibration, config, or code changes.
#
# Each update uses fresh rollouts and exactly one clipped replay. Branch 0 has
# zero actor advantage. Explorer credit is leave-one-out correctness plus an
# anchor-rescue bonus, with KL to the exact z=0 policy and a bounded-injection
# penalty.

# %%
metrics_path = RUN_DIR / "train" / "metrics.jsonl"
if PARTICLE_MODE == "learned":
    TOTAL_RL_HOURS = profile["rl_hours"]
    minimum_rl_hours, maximum_rl_hours = profile["rl_hours_range"]
    if not minimum_rl_hours <= TOTAL_RL_HOURS <= maximum_rl_hours:
        raise ValueError(
            f"This profile requires {minimum_rl_hours}–{maximum_rl_hours} RL hours."
        )
    stop_buffer_minutes = float(CONFIG_DATA["runtime"]["stop_buffer_minutes"])
    elapsed_seconds = 0.0
    if metrics_path.is_file() and metrics_path.stat().st_size:
        elapsed_seconds = float(
            json.loads(metrics_path.read_text().splitlines()[-1])["elapsed_seconds"]
        )
    remaining_seconds = TOTAL_RL_HOURS * 3600 - elapsed_seconds
    if remaining_seconds <= stop_buffer_minutes * 60:
        raise RuntimeError("The cumulative full-RL time budget is already exhausted.")
    deadline = time.time() + remaining_seconds - stop_buffer_minutes * 60
    print({
        "elapsed_rl_hours": elapsed_seconds / 3600,
        "remaining_rl_hours": remaining_seconds / 3600,
        "stop_buffer_minutes": stop_buffer_minutes,
    })
    rl_env = {**GPU_ENV, "HRM_V1_DEADLINE_UNIX": str(deadline)}

    command = [
        "torchrun", "--standalone", NPROC_ARG,
        "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "rl",
    ]
    last_file = RUN_DIR / "train" / "last.json"
    if last_file.is_file():
        last = json.loads(last_file.read_text())
        command += ["--resume-from", RUN_DIR / "train" / last["checkpoint"]]
    run(command, env=rl_env)
else:
    print("Gaussian mode: skipping PPO; only Q is trainable in this V1.")

# %%
if PARTICLE_MODE == "learned" and metrics_path.is_file():
    tail = metrics_path.read_text().splitlines()[-5:]
    for line in tail:
        print(json.dumps(json.loads(line), indent=2))

# %% [markdown]
# ## 10. Recalibrate Q on fresh frozen-policy rollouts
#
# Cached warmup states are not reused for the final selection claim. This stage
# regenerates only the two held-out
# partitions under the frozen inference policy, re-verifies every K=4 group, chooses
# separate math/code margins, and tests each fixed margin on its untouched
# safety partition. In Gaussian mode the "checkpoint" is the standalone Q
# safetensors file plus the checksummed frozen scale artifact; there is no actor
# checkpoint. In learned mode it is the final adapter/Q checkpoint. If either
# gate fails, that task uses branch 0.

# %%
# Refresh the provider semantic probe if the experiment has run for more than 48 hours.
run([
    PYTHON, "scripts/v1_code_provider_probe.py", "--config", CONFIG,
    "--output", RUN_DIR / "code-provider-canary.json",
])
run([
    "torchrun", "--standalone", NPROC_ARG,
    "scripts/v1_distributed.py", "--config", CONFIG,
    "--stage", "final_q_calibration",
] + inference_checkpoint_args(), env=GPU_ENV)
final_q = json.loads((RUN_DIR / "final-q-calibration.json").read_text())
print(json.dumps(final_q["gates"], indent=2))

# %% [markdown]
# ## 11. Three math evaluations
#
# Equal four-completion budgets are reported transparently:
#
# - clean greedy: particle branch 0;
# - ordinary sampling: four stochastic zero-latent responses, mean/oracle/majority;
# - matched control: that greedy anchor plus three zero-latent samples;
# - particles: one greedy anchor + three particle explorers, mean/oracle/Q-selected.
#
# Majority vote uses normalized parsed math answers and an earliest-sample tie
# break. Q uses the fresh, task-specific final-state gate above—never benchmark
# labels. Reports include paired
# bootstrap intervals for Q minus anchor and particle-oracle minus ordinary
# sampling-oracle; GSM-Symbolic resamples original templates rather than
# pretending five correlated perturbations are independent.

# %%
run([
    "torchrun", "--standalone", NPROC_ARG,
    "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "eval_math",
] + inference_checkpoint_args(), env=GPU_ENV)
math_summary = json.loads((RUN_DIR / "eval_math" / "summary.json").read_text())
print(json.dumps(math_summary["benchmarks"], indent=2))

# %% [markdown]
# ## 12. MBPP+ generation and sandboxed evaluation
#
# The GPU stage only writes candidate programs. It does **not** execute model
# output. Five files are produced: ordinary K=4, matched anchor+sample K=4,
# particle K=4, greedy anchor, and Q-selected particle. Code text majority voting is intentionally omitted;
# it has no semantic meaning.
#
# EvalPlus execution uses the official 0.3.1 Docker image pinned by its release
# SHA256 digest. Keep Docker isolation—do
# not replace this with `exec`, `pytest`, or `subprocess` on the training host.

# %%
run([
    "torchrun", "--standalone", NPROC_ARG,
    "scripts/v1_distributed.py", "--config", CONFIG, "--stage", "eval_mbpp",
] + inference_checkpoint_args(), env=GPU_ENV)
mbpp_dir = RUN_DIR / "evalplus_mbpp"
mbpp_generation = json.loads((mbpp_dir / "generation-summary.json").read_text())
print(json.dumps(mbpp_generation, indent=2))

# %%
RUN_EVALPLUS_DOCKER = False  # explicit opt-in; this is the only code-execution cell
if RUN_EVALPLUS_DOCKER:
    # EvalPlus otherwise reuses stale result JSON. Remove only the declared result
    # artifacts declared by this generation run; sample hashes remain fixed.
    for result_name in mbpp_generation["expected_result_files"]:
        (mbpp_dir / result_name).unlink(missing_ok=True)
    for shell_command in mbpp_generation["docker_commands"]:
        # Commands are generated by the checked-in runner and execute inside
        # ganler/evalplus, with only this result directory mounted.
        subprocess.run(shell_command, cwd=mbpp_dir, shell=True, check=True)
else:
    print("MBPP+ samples are ready. Set RUN_EVALPLUS_DOCKER=True to score them safely.")

# %%
if RUN_EVALPLUS_DOCKER:
    run([PYTHON, "scripts/v1_summarize_evalplus.py", "--directory", mbpp_dir])
    mbpp_summary = json.loads((mbpp_dir / "summary.json").read_text())
    print(json.dumps(mbpp_summary, indent=2))

# %% [markdown]
# ## Why LoRA is off in V1
#
# Generic LoRA changes branch 0, so it removes the exact pretrained-anchor
# guarantee and confounds whether gains came from H-state particles or ordinary
# fine-tuning. The default Gaussian V1 therefore trains only Q; its particle is
# a fixed random direction and its scale is selected before Q collection. The
# optional learned mode trains the particle adapter and Q while leaving the
# backbone frozen.
#
# If the code pilot is all-zero, a separate code-format/code-SFT LoRA experiment
# is reasonable, but it must be reported as a different baseline:
#
# 1. apply PEFT to the official `HrmTextForCausalLM` **before** the particle wrapper;
# 2. target reviewed H-stack modules (for example q/v/o and MLP down projections);
# 3. SFT it on a substantial verified code corpus;
# 4. freeze it before Q collection/RL and redefine that SFT policy as branch 0;
# 5. rerun every baseline and decontamination check.
#
# Direct Gaussian H-noise is the primary V1, not an ablation: it is the closest
# simple transfer of PTRM's mechanism and has no trainable actor. The optional
# learned particle differs because its query-conditioned, response-coherent
# direction is optimized by external correctness. Comparing the two modes later
# cleanly tests whether the learned adapter adds value beyond stochastic search.

# %% [markdown]
# ## Sources used by this notebook
#
# - [HRM-Text-1B model card](https://huggingface.co/sapientinc/HRM-Text-1B)
# - [Official HRM-Text implementation](https://github.com/sapientinc/HRM-Text)
# - [TRM: Less is More—Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
# - [PTRM paper](https://arxiv.org/abs/2605.19943)
# - [DAPO-Math-17k processed data](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)
# - [Verified/decontaminated Python problems](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled)
# - [Open-R1 remote code reward](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py)
# - [Math-Verify](https://github.com/huggingface/Math-Verify)
# - [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
# - [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
# - [GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic)
# - [EvalPlus / MBPP+](https://github.com/evalplus/evalplus)
# - [EvalPlus 0.3.1 release and pinned image](https://github.com/evalplus/evalplus/releases/tag/v0.3.1)
