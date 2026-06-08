# evals — coding benchmark suite

Lightweight pass@1 harness for the open coding-model lab. Every checkpoint
that comes out of training gets evaluated here before it ships.

## What's in the suite

| Eval | What it tests | Problems | Format |
|---|---|---|---|
| `humaneval` | Short Python functions from docstrings | 164 | pass@1 via test execution |
| `mbpp` | "Mostly Basic Python Problems", sanitized split | ~257 | pass@1 via assert execution |

The plan is to add `humanevalplus` (HumanEval+), `livecodebench` (contamination-free), and `bigcodebench` later. Keep this file in sync.

## Quick start

```bash
# Smoke test on the 0.5B reference model (MPS, ~1-2 min on M-series Mac)
python -m evals.run_baseline \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --device mps --limit 20

# Full suite on a real GPU
python -m evals.run_baseline \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --device cuda

# Eval a local checkpoint from continued pretraining
python -m evals.run_baseline \
  --model ./checkpoints/coding-model-001/step-10000 \
  --device cuda
```

Each run writes:
- `results/humaneval__<model>.jsonl` — raw samples (consumed by `human_eval` scorer)
- `results/humaneval__<model>.results.json` — per-problem pass/fail
- `results/mbpp__<model>.jsonl` — raw samples
- `results/mbpp__<model>.summary.json` — pass@1
- `results/report__<model>__<timestamp>.json` — combined run summary

## Why these evals and not others

- **HumanEval** is the lingua franca. Every coding model reports it. You can compare to any paper or leaderboard with a single number.
- **MBPP** is a different shape: shorter, more "real beginner problems." Tests whether the model can solve a problem stated in plain English, not just fill in a signature.
- **Both** together catch the common failure mode where a model is good at in-filling (HumanEval) but bad at problem → code translation (MBPP).

## What pass@1 is and is not

- **pass@1** = fraction of problems solved on the *first* sample, greedy decoding (temperature = 0).
- It rewards **calibrated, single-shot** code generation. Good for: autocomplete, code assistants.
- It does **not** reward: diversity, exploration, or long chain-of-thought. For that you'd want pass@10 or pass@100.
- This lab targets the assistant use case → pass@1 is the right headline metric.

## Adding a model to the leaderboard

After a successful eval, append a row to `coding-model-leaderboard.md` in the project root:

```md
| model | size | humaneval | mbpp | date | run_id |
|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct | 0.5B | 0.43 | 0.55 | 2026-06-07 | report__... |
```

The point of the leaderboard is to compare **your checkpoints** to reference models, not to compete with frontier labs. A small model that beats Qwen 2.5 Coder 0.5B at the same size on both is a real result.

## Adding a new eval

1. Drop a new file `evals/<name>.py` exposing a `run(model, device, limit, instruct, out_path) -> dict` function.
2. Add the name to `SUITE` in `run_baseline.py`.
3. Document the eval in this README.
4. Add a row to the leaderboard template.

Keep the runners thin: one model load, one generation loop, one scoring step. No abstractions until you have three of them.

## What this eval suite is *not*

- Not a replacement for `lm-evaluation-harness` or `bigcode-evaluation-harness`. Those are the standards; we use them as references but keep our own runner because (a) it works on MPS out of the box, (b) it logs results in the lab's format, and (c) it costs zero extra dependencies beyond `transformers` + `datasets` + `human-eval`.
- Not a contamination check. If you train on GitHub code, your model has seen HumanEval. Use `humanevalplus` and `livecodebench` for honest numbers.
- Not a deployment test. Real assistants need latency, cost, and safety evals too — those are a different folder, not this one.
