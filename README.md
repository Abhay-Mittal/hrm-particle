# HRM Gaussian H-state branching POC

This repository tests one narrow question:

> Can direct, response-coherent Gaussian perturbations between HRM's H1 and H2
> cycles make a frozen HRM-Text-1B produce correct alternatives that matched
> ordinary token sampling misses?

The default V1 fixes particle 0 as an exact `epsilon=0` anchor and gives each of
three explorers one normalized, hidden-sized Gaussian direction that is reused
for the full response. A predeclared development sweep selects a small fixed
noise scale on one half of its prompts and confirms it on the untouched half;
the Gaussian intervention has no trainable parameters. Only a detached Q head
is trained, using strict external-verifier correctness to select among
already-generated candidates. Q is **not** an actor reward.

An optional `particles.mode: learned` keeps the earlier query-conditioned
adapter and verifier-RL path as a follow-up. Keeping the modes separate makes
the V1 answer the simpler PTRM-style question before adding learned capacity.

This is a directional POC, not paper-level evidence. The older single-GPU
budget profiles use 600–800 training questions, `K=4`, an average target of 96
tokens, and a hard 128-token cap.

The larger mixed math/code V1 requested after that initial POC lives in
[`notebooks/HRM_Text_1B_Particle_V1.ipynb`](notebooks/HRM_Text_1B_Particle_V1.ipynb).
It is a control-plane notebook whose default is a restartable two-GPU POC: an
independent 128-prompt, two-half scale sweep, an 8,192-candidate correctness-Q warmup,
and matched K=4 evaluation on GSM8K, MATH-500, GSM-Symbolic, and MBPP+. The
3,000-prompt math/code pool supports scale selection and the optional learned
RL extension. The notebook does not check accelerator names; it requires CUDA,
native BF16, NCCL, and enough VRAM.

## What is implemented

- A parameter-free direct Gaussian H-state intervention, an optional
  query-conditioned learned adapter, and a shared correctness Q head.
- HRM recurrent-cache and injection hooks, fixed-width rollout/replay, token PPO,
  anchor-rescue credit, exact verification, prompt/PrefixLM formatting, and tests.
- Deterministic exact-arithmetic JSONL generation with train/dev/test/OOD
  isolation, held-out OOD templates, manifests, hashes, and private eval seeds.
- Optional strict-numeric Big-Math import with explicit source allowlists and
  benchmark-family contamination guards.
- Budget-checked single-GPU profiles, dry-run launchers, evaluation metrics, and an
  offline dummy optimizer smoke test.
- A BF16 two/three-rank DDP runner with FP32-master AdamW, empirical 75%-target
  VRAM batch planning, a checksummed Gaussian scale-selection artifact,
  binary-Q/shaped-actor reward
  separation, strict remote-only Python verification with per-batch sentinels,
  four-way source-stratified Q splits, fresh frozen-policy task-specific Q gates,
  Q-only Gaussian inference, optional learned-mode atomic checkpoints, and
  pinned/summarized EvalPlus Docker outputs.

## Direct Gaussian particles in plain language

HRM processes a token through a first high-level reasoning cycle (H1), then a
second one (H2). Gaussian mode adds a small random direction between them:

```text
one hidden-sized Gaussian direction epsilon_k (fixed for the response)
                              |
                              v
H1 state --> normalized alpha * RMS(H1) nudge --> H2 --> token logits
```

For explorer `k`, the direct update is approximately
`delta_k = alpha * RMS(H1) * epsilon_k / RMS(epsilon_k)`. The same `epsilon_k`
is reused at every response step, so this is a coherent stochastic trajectory,
not fresh token-by-token noise. Branch 0 uses an exactly zero direction; with a
frozen backbone it preserves the clean greedy anchor bit-for-bit.

The scale `alpha` is chosen once from `[.005, .01, .02, .03, .05]` on 128
unique `rl_train` prompts using common random directions and a scale-zero
control. The runner forms deterministic, disjoint 50/50 halves separately
within math and code, interleaving sources before assigning selection versus
confirmation. Selection chooses the smallest positive scale within the
configured tolerance of the best oracle pass@4 on the selection half. If every
positive scale is worse than zero there, the run aborts rather than forcing
noise. The chosen scale then faces a no-degradation gate on the untouched
confirmation half: its point-estimate oracle delta versus zero must be
nonnegative overall and separately for math and code. Only after that gate is
the scale frozen for Q collection and benchmarks.

The artifact's `development_evidence_positive` flag is computed only on the
untouched confirmation half: it requires the paired-bootstrap lower bound for
selected-scale oracle pass@4 minus scale zero to be positive. Confirmation
never reselects the scale. This confidence-interval flag is stronger than the
confirmation no-degradation gate and is recorded rather than required for this
POC. No Q or benchmark label participates in scale selection.

In optional learned mode, a 64-dimensional latent and the question summary feed
a bounded adapter. That mode can be optimized with verifier-only PPO, but it is
not the default V1 and should be reported separately.

## Important first-token semantics

The default configs use `first_token_mode: causal_prefix`. They perform one
clean, shared, fully bidirectional PrefixLM prompt prefill, then process a fixed
`"\nSolution:\n"` prefix as causal response tokens with particle injection.
Consequently, the H-state intervention affects the first **sampled** token without changing
the official prompt boundary. The fixed prefix is not sampled and receives no
policy loss.

Every ordinary-sampling/LoRA/soft-latent baseline must receive the identical
fixed response prefix and token accounting. Otherwise a prompt-format change is
confounded with the particle effect.

`shared_prefill` is a useful no-fixed-prefix diagnostic; there, injection begins
when sampled token 1 is fed back and can affect sampled token 2 onward.
`branch_recompute` is the expensive `K`-full-prompt ablation. Removing, retyping,
or appending `<|im_end|>` as a causal response token is not equivalent to the
checkpoint's clean PrefixLM prefill and is not a valid shortcut.

## Local quick start

Python 3.11+ is required. HRM-Text needs Transformers 5.9+ according to its
[official model card](https://huggingface.co/sapientinc/HRM-Text-1B).
The paid-run configs pin model revision
`9f082d68b8cd0ebc56e33f1c88c45609174c272c` rather than mutable `main`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,data,test]'

# No network or model weights: schema, verifier, Gaussian injection,
# optional adapter PPO, and detached Q update.
python scripts/run_smoke.py

# Full offline suite.
pytest -q
```

For V1, open the notebook above. Its first CPU-only cells install
`requirements-v1.txt`, prepare and checksum the public data, and run the full
test suite. It then launches real-model smoke tests and a distributed memory
probe that chooses the largest shared prompt batch near 75% VRAM (80% hard cap),
while adjusting batching for the detected devices. It then selects the Gaussian
scale, collects/verifies Q states, trains Q, and evaluates the frozen policy.

The notebook defaults to `GPU_COUNT = 2` and
`configs/v1_2gpu_poc.yaml`; set `GPU_COUNT = 3` to select the larger
`configs/v1_3gpu_full.yaml` profile. Do not edit `torchrun` ranks separately:
the notebook derives them from `GPU_COUNT`. The memory probe measures usable
bytes on every rank, pads its longest-prompt test to the full response-token
cap, and uses the smallest safe batch across the cluster. Accelerator names are
reported for provenance only and never enter the decision.

These are separate budget profiles, not a controlled two-versus-three-GPU
comparison. Q collection and evaluation shard prompt groups across every rank;
optional learned-mode optimizer batches and RNG streams intentionally cannot
resume across profiles with different world sizes.

The JSONL `prompt` field is a raw math question. Do not manually add HRM special
tokens. The rollout formatter applies the official composite `synth,cot`
condition and `token_type_ids` PrefixLM boundary.

## Prepare reproducible data

This bundle already includes `data/processed` at 800/200/400/400 so the full
pipeline can run immediately. Its dev/test/OOD rows use the documented public
smoke seed: they are suitable for end-to-end plumbing and preliminary training,
but **not private holdouts and not paper evidence**. Regenerate them with your
own private seed before reporting any result.

Create one private evaluation seed and keep it out of version control:

```bash
python scripts/prepare_data.py make-eval-seed
python scripts/prepare_data.py synthetic \
  --eval-seed-file secrets/eval_seed.txt \
  --train-count 800 --dev-count 200 --test-count 400 --ood-count 400
python scripts/prepare_data.py validate --data-dir data/processed
```

The raw secret is never written to the dataset; the manifest stores a one-way
fingerprint. Train/dev/test share problem families but use independent derived
seeds. OOD uses an exclusive template partition. Validation rejects duplicate
IDs, prompts, semantic operand sets, wrong template partitions, count changes,
and checksum changes.

`data/sample/sample.jsonl` contains six schema fixtures only. Do not train on it.

### Optional Big-Math training rows

This is optional for the first synthetic POC. The importer supports both the
raw `problem`/`answer` schema and Open-R1's processed `prompt`/`solution` schema.
It accepts only answers handled by the strict numeric verifier. Sources must be
explicitly allow-listed; common evaluation families such as MATH, GSM8K,
Olympiads, AOPS, HARP, and Omni-MATH are refused.

```bash
pip install -r requirements-data.txt
python scripts/prepare_data.py bigmath \
  --dataset-config all \
  --source big_math --source orca_math \
  --min-solve-rate 0.1 --max-solve-rate 0.8 \
  --limit 800 --max-rows-scanned 100000
```

The result is written separately to `data/external/big_math_train.jsonl`; the
tool never appends it to synthetic train or evaluation files. Review the
[processed dataset card](https://huggingface.co/datasets/open-r1/Big-Math-RL-Verified-Processed)
and upstream [Big-Math card](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified)
before use.

Import uses deterministic reservoir sampling and never scans beyond
`--max-rows-scanned`; this bounds time and memory even for a streaming or
unbounded source. By default it fails rather than silently returning fewer than
`--limit`. Use `--allow-fewer` only after reviewing the reported filter counts.

## Train and evaluate

First validate everything without loading the model:

```bash
python scripts/train.py --config configs/poc_h100.yaml --dry-run
```

Then, on the rented GPU:

```bash
python scripts/train.py \
  --config configs/poc_h100.yaml \
  --max-cost-usd 40 --max-gpu-hours 12

python scripts/evaluate.py \
  --config configs/eval.yaml \
  --checkpoint runs/poc_h100/checkpoint-last.pt
```

CLI budget overrides may only lower checked-in ceilings. The launcher exports a
wall-clock deadline for the trainer, but this cannot terminate the Runpod pod or
guarantee provider billing. Set a provider-side cap/alarm, copy results off, and
stop the pod immediately after the process finishes. See [docs/RUNPOD.md](docs/RUNPOD.md).

## Decision metrics

The evaluation compares the same total candidate count and reports:

- exact anchor accuracy and mean explorer accuracy;
- oracle pass@4;
- rescue@3 conditional on the exact anchor being wrong;
- Q-selected accuracy and regret to oracle;
- within-prompt Q pairwise accuracy;
- response length and format validity;
- prompt-paired bootstrap confidence intervals on test and held-template OOD.

The budgeted evaluator makes one primary matched comparison at temperature 0.8
and top-p 1.0; it does not tune on final benchmarks. The only scale sweep is the
predeclared `rl_train` development stage. Its task-balanced, source-interleaved
selection/confirmation IDs, common RNG seed, per-scale outcomes, selection rule, confirmation-only
no-degradation result, stronger evidence flag, and artifact hash are retained.
The
mechanism has initial headroom only if Gaussian oracle/rescue beats ordinary
zero-latent sampling at the same candidate budget.

Full-softmax sampling is also important in optional learned mode: an explorer
token must have finite support under the zero-latent PPO reference. Independent
dynamic nucleus masks can otherwise create pathological KL. A temperature/top-p
frontier, response-start soft-latent control, learned-adapter RL, and ordinary
LoRA RLVR (each with a comparable selector) are follow-up experiments required
before an H-specific claim.

## Safety and reproducibility rules

- Keep the backbone, LM head, and any format LoRA frozen in POC-A.
- Treat Gaussian scale selection as development-set model selection: select on
  the task-balanced, source-interleaved selection half; require overall,
  math-only, and code-only no-degradation without reselection on confirmation;
  record CI-positive confirmation as stronger evidence; and never use Q
  warmup/calibration or final benchmark labels to choose it.
- In optional learned mode, actor rewards come only from the external verifier.
- Detach terminal/query states before Q supervision reaches the actor.
- Store Gaussian directions, masks, token RNG state, the selected scale, and
  artifact hashes. Learned mode additionally stores old log-probabilities for
  exact replay; pre-update PPO ratios should be 1.
- Never report the public smoke seed or checked-in sample as private evaluation.
- Do not selectively retry/drop all-wrong rollout groups.
- Record the model revision, config, dataset manifest, package versions, GPU,
  seed fingerprints, and checkpoint hash with every result.

## License

Project code is Apache-2.0. Model and optional datasets are downloaded
separately and retain their upstream terms; see `NOTICE`.
