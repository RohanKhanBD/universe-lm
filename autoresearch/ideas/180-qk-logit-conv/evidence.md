# Evidence — 180 qk-logit-conv

## r1 — 2026-06-15T06:16:53Z — daemon bounced → needs-recode
- **Failure**: `build-smoke failed: SMOKE_FAIL: ImportError: cannot import name 'Tiny1M3MLogitConvConfig' from 'configs.llm_config' (/root/universe-lm/configs/llm_config.py)`
- **Cause**: the box's view of `configs/llm_config.py` did not yet contain `Tiny1M3MLogitConvConfig` at the moment the daemon ran the build-smoke. The class is defined locally in `configs/llm_config.py:6206` with `use_logit_conv: bool = True` and imports cleanly, so this was a sync-window issue between local commit and box pick-up, not a missing-code bug.

## r1 — 2026-06-15T06:25:43Z — recode round 1
- **Action**: re-verify the full claim chain.
  - `from configs.llm_config import Tiny1M3MLogitConvConfig` → imports cleanly.
  - `PYTHONPATH=. python autoresearch/bin/_box_smoke.py _arq_180-qk-logit-conv.py` → `SMOKE_OK` (CPU construct succeeds; the same script the daemon runs on the box).
  - **Step-0 byte-identical check** (treatment vs baseline, same seed-42 weights, `x = randint(0, vocab, (1, 32))`):
    - `max-abs-diff = 2.608e-08` (fp32 reduction-order noise; functionally 0.0). Well below the 1e-5 threshold from review.md.
  - **Conv delta-init unit check**: `logit_conv_w[:, K-1] = 1.0`, rest 0 ⇒ `conv(scores) - scores = 0` byte-exact across all 4 heads, all 12 blocks (param tensor inspected post-construction).
- **Files**: no code changes — implementation was already correct; the box just needs a fresh pick-up of the current HEAD.
- **Release**: `flip.sh 180-qk-logit-conv needs-run code-impl "<fix summary>" 2` — round 2.
