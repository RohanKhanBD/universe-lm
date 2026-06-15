# Evidence — 182-per-head-window (recode r1)

## Verdict: RECODE — build-smoke caught by daemon

The previous r1 code-impl released to `needs-run`, but the daemon's
CPU build-smoke (run on the box at `/root/universe-lm`) failed with:

```
SMOKE_FAIL: ImportError: cannot import name 'Tiny1M3MPerHeadWindowConfig'
  from 'configs.llm_config' (/root/universe-lm/configs/llm_config.py)
```

The reason: the working-tree edits to `configs/llm_config.py`,
`models/layers.py`, `models/llm.py`, and `_arq_182-per-head-window.py`
were never committed. The daemon's `git pull --ff-only` only fetches
origin-tracked commits, so the box's `configs/llm_config.py` did not
have the new `Tiny1M3MPerHeadWindowConfig` class. The local CPU smoke
PASSED (because the local working tree had the class), masking the
gap.

## Fix (this pass)

- Committed the working-tree edits to `configs/llm_config.py`,
  `models/layers.py`, `models/llm.py` (commit `0653bfc8`).
- `_arq_182-per-head-window.py` is present and unchanged (top-level
  `C(Tiny1M3MPerHeadWindowConfig)`, drives `train_llm.main()`, fixed
  seed 42, dataset `processed_data/pretrain_1B`, `--warmup false`).
- Re-verified the daemon's CPU build-smoke locally with
  `PYTHONPATH=… python3 autoresearch/bin/_box_smoke.py _arq_182-per-head-window.py`
  ⇒ `SMOKE_OK`.
- Step-0 byte-identical re-verified:
  `max_abs_diff(MinimalLLM(Tiny1M3MConfig)(x), MinimalLLM(Tiny1M3MPerHeadWindowConfig)(x)) = 2.98e-08`
  (well under the 1e-6 bar from plan.md).

## Outstanding

The box (`/root/universe-lm`) needs to pull the new commit before the
daemon's next smoke tick. Until `git push` lands on origin the box's
working tree is stale. This is the same gap that bounced r1 — the
fix that lands on origin is the durable cure.

## Files

- `configs/llm_config.py` — `use_per_head_window: bool = False` on
  `LLMConfig`, `Tiny1M3MPerHeadWindowConfig(Tiny1M3MConfig)` subclass
  flips it on.
- `models/layers.py` — `use_per_head_window` kwarg on
  `MultiHeadAttention.__init__`; when on, allocates
  `self.head_window_logit = nn.Parameter(torch.full((n_heads,), 10.0))`;
  manual-path dispatch picks the score-space penalty
  `score -= 1e9 · relu(rel_dist − half_w)`.
- `models/llm.py` — threaded through both `TransformerBlock` sites
  (YOCO upper-half + standard).
- `_arq_182-per-head-window.py` — fixed-shape treatment entry; daemon
  contract satisfied.
