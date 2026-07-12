# Data layout

`sample/sample.jsonl` is a six-row, checked-in schema fixture. It is for smoke
tests only, not training or reporting results.

`processed/` is bundled at the planned 800 train / 200 dev / 400 test / 400 OOD
scale. It uses `--public-eval-seed-for-smoke`, so it is ready for end-to-end POC
plumbing and preliminary training, but its evaluation rows are public smoke
holdouts—not private evidence. Regenerate in place with a private seed before
reporting results. `processed/DATA_STATUS.txt` is atomically updated by the CLI
so the current status remains visible after regeneration.

Regenerate private evaluation files locally:

```text
data/processed/
  manifest.json
  train.jsonl       # public training seed, in-domain templates
  dev.jsonl         # private eval seed, in-domain templates
  test.jsonl        # private eval seed, in-domain templates
  ood.jsonl         # private eval seed, held-out templates

data/external/
  big_math_train.jsonl               # optional, external training only
  big_math_train.jsonl.manifest.json
```

The manifest has counts and SHA-256 checksums. `prepare_data.py validate`
recomputes them and rejects prompt, ID, semantic-problem, and template leakage.
Raw seeds are not recorded; only one-way fingerprints are stored.

Each JSONL object contains:

- `schema_version`, `id`, and `split`;
- `family`, `template_id`, `prompt`, and canonical `answer`;
- `answer_type` (`integer` or `rational`) and `verifier`;
- `source`, `difficulty`, and JSON-only `metadata`.

The exact verifier accepts integers, reduced/unreduced fractions, finite decimals,
percentages, simple LaTeX fractions, `<answer>...</answer>`, `#### ...`, and balanced
`\boxed{...}` output. It never evaluates arbitrary code or symbolic expressions.
