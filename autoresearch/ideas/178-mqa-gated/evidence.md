# Evidence — 178-mqa-gated

## r3 — needs-recode (build-smoke: same ImportError, root cause unchanged)

**Cause (daemon log 2026-06-15T06:29:42Z):**
`SMOKE_FAIL: ImportError: cannot import name 'Tiny1M3MMQAGatedConfig' from 'configs.llm_config' (/root/universe-lm/configs/llm_config.py)`. Identical failure to r1 and r2 — the box's `git pull --ff-only` from `origin/orchestrate-codex-fallback` brings in no new commits because origin's HEAD is at `3a449a2` (predates `0653bfc` by ~30 minutes). The implementation commits (`0653bfc` wiring + `cbb3968` stub) live only on the local `orchestrate-codex-fallback` branch; the box cannot see them without a user-driven push.

**Code state (verified r3):**
- Local tree at `35af083` (current HEAD), 70 commits ahead of origin.
- `configs/llm_config.py:6273` defines `class Tiny1M3MMQAGatedConfig(Tiny1M3MConfig): use_mqa_gated: bool = True` — present.
- `models/layers.py` allocates zero-init `W_K_shared`, `W_V_shared`, `mqa_gate_k`, `mqa_gate_v` per block when flag is on — present.
- `models/llm.py` threads `use_mqa_gated` through `MinimalLLM` → both `TransformerBlock` sites — present.
- `_arq_178-mqa-gated.py` defines top-level `C(Tiny1M3MMQAGatedConfig)` and dispatches `train_llm.main()` with the required flags — present.

**Self-check (re-verified r3):**
```
$ PYTHONPATH=. python3 autoresearch/bin/_box_smoke.py _arq_178-mqa-gated.py
SMOKE_OK
```
Stub imports cleanly, `C()` constructs, `MinimalLLM(C())` builds on CPU. Step-0 byte-identity preserved (β=0 + zero-init shared K/V ⇒ `max-abs-diff = 0.0` vs no-flag baseline at seed 42).

**Why this is r3 and not another r2-style "commit and release":**
The implementation is already on disk and committed. The previous worker (round 2) noted in plan.md that the only thing keeping the box from running the treatment is a missing push from the user. Recoding the code again would not change the box-side state. The fix is one human action: `git push origin orchestrate-codex-fallback` from the local machine. Until that push lands, every daemon tick that claims this idea will hit the same ImportError and bounce it back.

**What I did NOT do (and why):**
- **No push.** The recode protocol is "no auto-push — commit locally only; wait for human review before any push." Per `feedback-dont-push-without-approval`. The previous worker observed the same constraint; this pass observes it identically.
- **No runtime monkey-patch in the stub.** I considered patching `models.layers.MultiHeadAttention` at runtime inside `_arq_178-mqa-gated.py` to bypass the box-stale `configs/llm_config.py` / `models/layers.py`. Rejected because (a) it would not exercise the gate branch even if smoke passes — `getattr(cfg, "use_mqa_gated", False)` returns False when the field is absent on the box's `Tiny1M3MConfig`, so the treatment silently degenerates to a baseline run; (b) it duplicates the implementation logic and is brittle to upstream refactors; (c) it would silently produce a null verdict and defeat the probe's primary signal.
- **No `flip.sh … needs-review … "blocked: <why>"` bounce.** The recode prompt offers this as a blocker escape valve, but the implementation is not blocked — the code is correct and committed. The blocker is the human-push coordination step, which is outside the recode agent's loop. The protocol's documented behavior on a fix-correct-but-not-pushed is "commit and release," and the daemon's recode-budget cap (`MAX_RECODE_ROUNDS=3`) handles the "user never pushes" case by auto-closing the axis to `rejected`.

**Round status:**
- This is round 3 (current `round: 2`, bumping to 3 on release).
- `flip.sh`'s cap check fires on the next `needs-recode` flip; with round=3 in the frontmatter, that bounce auto-closes 178 to `rejected` (and writes "exhausted 3 recode rounds, axis abandoned" to `closed.md`). That's the documented terminal state for an axis that can't stabilize — the cap mechanism is doing what it's designed for if the user does not push in time.

**Human action required:**
```
git push origin orchestrate-codex-fallback
```
This advances origin's HEAD past `0653bfc` so the box's `git pull --ff-only` brings in `Tiny1M3MMQAGatedConfig` (and the model wiring). After the push lands, the daemon's next tick will pass smoke and run the treatment at seed 42.
