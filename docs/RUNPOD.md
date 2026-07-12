# Runpod checklist for the $50–100 POC

This plan deliberately separates free/local correctness work from paid GPU
work. Do not rent a GPU until the complete offline suite passes.

## Cost envelope

The rates below are editable planning inputs, not guaranteed provider quotes.
Check the live Runpod rate immediately before launch and update `hourly_usd` in
the YAML. The launcher refuses a plan where `hourly_usd * max_gpu_hours` exceeds
`max_cost_usd`.

| Config | Data | Checked-in time ceiling | Planning rate | Training ceiling |
|---|---:|---:|---:|---:|
| `poc_h100.yaml` | 800 train | 12 h | $2.89/h | $40 |
| `poc_a100.yaml` | 600 train | 20 h | $1.39/h | $30 |

Reserve roughly $10–20 for final test/OOD generation and one recovery attempt.
Spend the remaining budget on a shortened soft-latent or ordinary-sampling
control only after the primary run shows oracle headroom. This is a directional
screen; three seeds and a complete control suite need a later budget.

The in-process limit cannot stop provider billing. Configure any provider-side
spend cap/auto-stop available to your account, set a phone/calendar alarm, and
stop the pod after copying artifacts. A finished or crashed process can leave a
pod charging.

## 1. Validate locally before renting

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,data,test]'
python scripts/run_smoke.py --require-full-hook
pytest -q
```

The default tests are offline. They should cover:

- exact `z=0` anchor identity and prompt-mask invariance;
- bounded nonzero explorer injection and gradients;
- Q/actor gradient separation;
- recurrent cache slot addressing and branch isolation;
- cached-vs-full replay and PPO ratio parity;
- first-token/PrefixLM masks;
- deterministic data, split leakage, and verifier behavior;
- a tiny dummy rollout and optimizer update.

Do not proceed if the full dummy hook is unavailable or any test is skipped for
an unexpected reason.

## 2. Create the pod

Use one H100 80 GB or A100 80 GB with:

- a current PyTorch/CUDA image;
- at least 80 GB persistent storage for weights, caches, checkpoints, and logs;
- enough shared memory (`--ipc=host`/large `/dev/shm` in a custom container);
- SSH access or Runpod's terminal;
- a provider-side timeout/spend alarm if available.

The model card currently requires Python-compatible `transformers>=5.9`. The
project requires Python 3.11+. Do not use a 40 GB card for the first attempt:
HRM's recurrent KV slots and `K=4` branches make memory behavior less forgiving
than an ordinary 1B transformer.

The configs pin Hugging Face revision
`9f082d68b8cd0ebc56e33f1c88c45609174c272c`; keep that value in the experiment
record if you intentionally update it later.

## 3. Copy and install

Copy this directory to persistent storage, then from its root:

```bash
python --version
nvidia-smi
python -m pip install --upgrade pip
pip install -e '.[train,data,test]'
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print(torch.cuda.get_device_name())"
python scripts/run_smoke.py --require-full-hook
pytest -q
```

Do not paste a Hugging Face token into source, YAML, shell history, or logs. Use
Runpod's secret/environment mechanism to set `HF_TOKEN` if your account needs it.
The code downloads `sapientinc/HRM-Text-1B` at runtime.

## 4. Generate sealed data on persistent storage

The copied project includes a complete 800/200/400/400 public-smoke dataset for
immediate plumbing. Do not treat its dev/test/OOD scores as unseen evidence.
Run the commands below with a new private seed before reporting results; files
are validated and atomically replaced without deleting the directory.

```bash
python scripts/prepare_data.py make-eval-seed
python scripts/prepare_data.py synthetic \
  --eval-seed-file secrets/eval_seed.txt \
  --train-count 800 --dev-count 200 --test-count 400 --ood-count 400
python scripts/prepare_data.py validate --data-dir data/processed
```

Back up `secrets/eval_seed.txt` securely if exact regeneration matters, but do
not copy it into a public repository or training artifact. The training code
reads only `train.jsonl`; test and OOD remain sealed until evaluation.

If using A100 and the checked-in 600-example plan, generating 800 rows is fine:
the config caps the number consumed. To reduce storage/noise, set train count to
600 instead.

Optional external training rows are kept in a separate namespace:

```bash
python scripts/prepare_data.py bigmath \
  --dataset-config all \
  --source big_math --source orca_math \
  --min-solve-rate 0.1 --max-solve-rate 0.8 \
  --limit 800 --max-rows-scanned 100000
```

Do not merge external rows into `test.jsonl` or `ood.jsonl`. The POC can be run
without Big-Math, which is preferable for the cheapest mechanism test.

## 5. Run paid preflight

```bash
python scripts/train.py --config configs/poc_h100.yaml --dry-run
```

For A100, substitute `configs/poc_a100.yaml`. Confirm the printed:

- dataset counts and checksum validation;
- output directory;
- particle count 4 and token cap 128;
- projected cost at or below the ceiling;
- frozen backbone and exact-zero adapter flags.

Run one tiny real-model batch before the full command if the trainer exposes a
step limit:

```bash
python scripts/train.py --config configs/poc_h100.yaml \
  --output-dir runs/preflight \
  --max-cost-usd 5 --max-gpu-hours 1 \
  --set data.max_train_examples=8 \
  --set optimization.rollout_rounds=1 \
  --set checkpointing.save_every_updates=1
```

Inspect GPU memory, loss finiteness, anchor equality, nonzero explorer deltas,
pre-update PPO ratio, verifier labels, and Q detachment before scaling up.

## 6. Launch the primary run

Use `tmux` so an SSH disconnect does not kill the process:

```bash
tmux new -s hrm-poc
python scripts/train.py \
  --config configs/poc_h100.yaml \
  --max-cost-usd 40 --max-gpu-hours 12
```

For A100:

```bash
python scripts/train.py \
  --config configs/poc_a100.yaml \
  --max-cost-usd 30 --max-gpu-hours 20
```

The CLI only permits budget overrides that tighten the YAML ceiling. It exports
`HRM_PARTICLE_DEADLINE_UNIX`, `HRM_PARTICLE_MAX_COST_USD`, and the hourly rate;
the trainer checks the deadline and should checkpoint before exiting.

Monitor periodically:

```bash
nvidia-smi
du -sh runs data ~/.cache/huggingface 2>/dev/null
```

Stop early if:

- CUDA memory is close to exhaustion before steady state;
- losses or relative injection RMS are non-finite;
- `z=0` logits differ from the frozen anchor;
- Q gradients reach adapter/backbone parameters;
- almost every group is all-correct or all-wrong (no useful RL signal);
- response lengths sit at the 128-token cap;
- throughput implies the provider ceiling will be crossed.

The primary training/evaluation configs intentionally use `top_p: 1.0` at
temperature 0.8. PPO compares particle log-probabilities with a zero-latent
reference. Separate top-p masks can give a particle-sampled token zero/masked
support under that reference, making KL pathological. Reserve top-p 0.95 for a
later inference-only decoding ablation after training.

Resume only from a complete checkpoint:

```bash
python scripts/train.py \
  --config configs/poc_h100.yaml \
  --resume-from runs/poc_h100/checkpoint-last.pt \
  --max-cost-usd 20 --max-gpu-hours 6
```

The reduced values are a new session ceiling, not accounting software for money
already spent. Track total provider charges separately.

## 7. Evaluate once, then copy artifacts

Run the sealed splits after training:

```bash
python scripts/evaluate.py \
  --config configs/eval.yaml \
  --checkpoint runs/poc_h100/checkpoint-last.pt
```

Evaluation should save aggregate metrics plus per-prompt candidate/reward/Q data
needed to recompute paired bootstrap intervals. Before stopping the pod, copy:

- adapter and Q checkpoints;
- the final checkpoint (which embeds the resolved config);
- synthetic manifest (not the private raw seed for a public artifact);
- `metrics.jsonl` and any terminal logs you explicitly captured;
- evaluation JSON/JSONL and checkpoint hashes.

Then verify the copied files and **stop the pod**.

The launcher does not capture a package/GPU snapshot automatically. Before
deleting the pod, separately save the output of `python -m pip freeze` and
`nvidia-smi` if you need exact environment provenance.

## First-token choice

Keep `causal_prefix` for the budgeted primary run. It uses one correct clean
bidirectional PrefixLM prompt prefill, then particle-injects the fixed causal
`"\nSolution:\n"` response prefix, so the adapter affects the first sampled
token. Give every baseline the same fixed prefix. `shared_prefill` is the
no-fixed-prefix diagnostic (adapter effect starts at sampled token 2), and
`branch_recompute` repeats full-prompt computation for every branch.

Do not simulate a response boundary by dropping or causally appending
`<|im_end|>`. That changes the PrefixLM computation the checkpoint was trained
to receive and confounds the result.
