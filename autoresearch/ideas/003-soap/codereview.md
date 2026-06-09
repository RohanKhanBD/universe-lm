# Code-review log — 003 soap

## r3 — 2026-06-09 — verdict: accept

**Round 3 cap — forced call. Blocking finding from r2 (3-seed protocol) is fixed. Code unchanged from r2 (the r2 fix was a doc fix, not a code fix; the optimizer body and routing were already correct). Accepting to `needs-run` for the screen20m A/B.**

**Re-check of r2 findings:**

- **🔴 3-seed protocol (r2 blocking) — FIXED.** `idea.md:39` now reads: "Seed 42, single seed, per the pipeline hard rule. |Δ| ≤ 0.05 is the noise band; a sub-noise result is logged inconclusive, not re-seeded." `plan.md` Step 2 renamed from "seed escalation" to "verdict (single seed, per pipeline hard rule)" and re-commits the same single-seed commitment. ✓
- **Pre-flight wall-clock tension (r2 minor) — RESOLVED.** `plan.md` §Run Step 0 now commits to 10 steps × ~3 min/step ≈ 30-35 min (not 100 × 5). The implementer also notes the original 100 × 5 was off by ~2× and acknowledges `use_soap_fp32_only` is somewhat moot since `state["L"]` is already fp32 unconditionally. The pre-flight is still a hard gate; the wall-clock is now honest. ✓
- **HP-default triplication (r2 minor) — NOT ADDRESSED.** `configs/llm_config.py:379` still hardcodes `use_soap_precondition_freq: int = 10`; `training/trainer.py:207-208` still has the `getattr` default; `optimizers/soap.py:49` still has the constructor default. Three sources of truth for one constant. **Not blocking** — the r2 review said so explicitly. Flag for a future cleanup PR.

**Identity re-check (didn't re-read code, but no code changed):**
- `optimizers/soap.py` — eigenbasis init to identity on step 0 → step 1 bit-identical to AdamW. ✓
- `training/trainer.py:200-211` — SOAP optimizer instantiated only when `use_soap=True AND soap_params` is non-empty. With `use_soap=False` (default), the SOAP branch is unreachable. ✓

**Verdict:** `accept`. Status → `needs-run`. The pre-flight is the only thing standing between this and a full screen20m A/B; the implementer is the right agent to run the pre-flight when ready.

## r2 — 2026-06-09 — verdict: revise

**Code is faithful and identity-safe, but the plan/idea both carry a 3-seed escalation protocol that violates the project's hard rule.**

**What I checked:**
- `optimizers/soap.py` (~186 LoC, within the <200 LoC budget for mined ideas).
- `configs/llm_config.py:378-379` — `use_soap: bool = False` and `use_soap_precondition_freq: int = 10`, on the line after `use_cautious_adamw`, per the r1 spec.
- `training/trainer.py:80-213` — SOAP routing added with a proper `getattr(config, "use_soap", False)` gate, `soap_params` list, and SOAP optimizer instantiated only when both flag is True AND `soap_params` is non-empty. Identity-safe: with `use_soap=False` the SOAP branch is unreachable.
- Mutual exclusion with the other routing levers: `muon_for_1d_norm`, `muon_for_embed`, `muon_for_output` set `is_muon_candidate=True`, which the SOAP branch (line 131: `not is_muon_candidate`) correctly respects. SOAP and Muon never both claim the same param.
- Eigenbasis init at identity (lines 155-158) → step 1 with `use_soap=True` is bit-identical to AdamW: `Q_L^T G Q_R = G` (Q=I) → projected grad = raw grad → standard Adam direction. ✓
- The `_adamw_step` 1D fallback in soap.py is dead code in the training path (trainer only routes 2D params to `soap_params`), but it's a defensive fallback for direct users of the optimizer class — acceptable.
- `optimizers/__init__.py:3,5` — `from .soap import SOAP` and `SOAP` in `__all__`. ✓
- `models/layers.py`, `models/llm.py` — UNTOUCHED. ✓ (per the parallel-AI coordination rule, I did not edit these)

**Findings:**

- **🔴 REVISE-BLOCKING — 3-seed protocol violates the ONE SEED ONLY rule.** `idea.md` §"Seed protocol" says: "3 seeds (42/43/44) when |Δ| ≤ 0.03 ... If single-seed pass and |Δ| > 0.03, ship the single seed. If single-seed pass and |Δ| ≤ 0.03, run the other two seeds before promoting to plan.md." `plan.md` §"Step 2 — seed escalation" carries the same protocol verbatim. **PIPELINE.md hard rule:** "Every ablation runs at a **single fixed seed (42)** — never multi-seed. No `≥3 seeds`, no seed sweeps, no per-seed means. ... Any idea, plan, or review that asks for more than one seed is **malformed** — strip it down to seed 42 instead. Read a sub-noise effect as **inconclusive, not real**." **Fix:** delete the seed-escalation paragraph from both `idea.md` and `plan.md`. Replace with a single sentence: "Seed 42, single seed, per the pipeline hard rule. |Δ| ≤ 0.05 is the noise band; a sub-noise result is logged inconclusive, not re-seeded." This is the only verdict-blocking finding.

- **Plan §Cost vs §Run pre-flight — internal tension (non-blocking, but should be reconciled).** §Cost estimates "eigh on 49152×49152 fp32 is ~3min/step and 18GB" and flags it as a likely bottleneck for `token_embedding`. §Run Step 0 commits to a "≤5 min wall-clock" pre-flight of 100 steps. 100 steps × 3 min/step = 300 min, not 5. The plan acknowledges a fp32-only fallback ("Will fall back to fp32-only for token_embedding if this becomes a bottleneck. Pre-flight catches this.") but no such flag exists in `configs/llm_config.py` or the SOAP code. **Fix:** either (a) shorten the pre-flight to 10 steps (still 30 min, not 5) and update the wall-clock estimate, or (b) add a `use_soap_fp32_only: bool = False` flag and code path that stores `L`/`Q_L` in fp32 only (which is what the code already does — `state["L"]` is fp32 unconditionally; the actual fallback is whether the *param* is fp32). State which. Not blocking; the pre-flight is the real gate and it will surface the bottleneck.

- **Pre-existing flag conflict (informational).** `use_soap_precondition_freq: int = 10` is on the same flag block as `use_soap: bool = False`. The default value of 10 is hardcoded in `configs/llm_config.py:379` and passed through `trainer.py:207-208` via `getattr(config, "use_soap_precondition_freq", 10)`. The SOAP `__init__` also has `precondition_frequency=10` as a default. Three sources of truth for the same constant — pick one (the config flag is the right one; the other two defaults should match it and be removed if the flag is the canonical source). Not blocking; flag the implementer to dedup.

**Plan ↔ idea consistency check:**
- pass bar: plan ≤ 4.5887, idea ≤ 4.5887 ✓
- fail: plan > 4.6364, idea > 4.6364 ✓
- noise: plan |Δ| ≤ 0.05, idea |Δ| ≤ 0.05 ✓
- control: V+q+SWA+HighRoPE on both ✓
- tier: screen20m on both ✓

**Coordination check:**
- `git diff` shows no stomp of the parallel-Claude's unstaged edits (the untracked `007-sigmoid-loss/`, `prompts/runner.md`, and the diff in `configs/llm_config.py` for `use_retention` are coherent with the 004 PR, not stomped).

**Hand-off to code-implementer:** the optimizer body is correct and the routing is clean. Delete the 3-seed protocol from both `idea.md` and `plan.md` (the only blocking fix). Reconcile the pre-flight wall-clock estimate or add the `use_soap_fp32_only` flag. Re-submit and re-review (round 3, the cap — next pass must be `accept` or `reject`).

## r1 — 2026-06-08 — verdict: revise
(implementation review — code was not yet at this round; placeholder for completeness)

## r0 — (none)
