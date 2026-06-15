# Champion promotion protocol — screen at 1 seed, confirm at 3

> **Status: policy (2026-06-15).** Amends the historical "🔴 ONE SEED ONLY" rule
> in [`PIPELINE.md`](PIPELINE.md). One seed is still the law for **screening**
> (the cheap funnel that runs every idea). **Promotion** — crowning a new
> champion, a rare and *irreversible* event that every future experiment then
> stacks on — is the one place that earns three seeds.

## Why a confirm gate exists

A champion is not just a result; it is the **baseline pinned in
`baseline-cache.json` that every later experiment is judged against and built on
top of** (see [`BASELINE-CACHE-DESIGN.md`](BASELINE-CACHE-DESIGN.md) and
`champion.json`). Promote a noise fluctuation and you don't just record a wrong
number — you poison the bar for the entire lineage after it. This has already
happened twice (`champion.json → rejected_promotions`):

- **180-qk-logit-conv** — a causal-mask leak (val 0.984) auto-promoted before the
  leak guard existed.
- **209-canon-conv** — a single-seed 6.2519 judged against a *base* control mean
  (6.3988) instead of the alibi champion (6.2403); a NULL promoted as a WIN.

Screening can be cheap and permissive because a NULL costs nothing. Promotion
must be strict because it is sticky. Different stakes ⇒ different seed budgets.

## The two stages

### Stage 1 — SCREEN (1 seed, band 0.04) — unchanged
Every idea runs once at **seed 42**, treatment-only, judged against the pinned
champion val. The daemon's existing gate:

```
candidate WIN  ⟺  trt_val < champion_val − 0.04
```

A result inside the band is **NULL / inconclusive** — logged to `closed.md`,
never promoted, never "confirmed with more seeds." Most ideas end here. **Do not
promote off a Stage-1 result.**

> ⚠️ **The 0.04 band is wrong for screening — it is cross-box drift, not paired
> noise.** Measured 2026-06-15 from all 21 ctrl runs in `remote-results/`:
> - **within-session** (same box, same day, fixed seed/data): 1σ ≈ **0.017**, 2σ ≈ 0.033
> - **cross-day / cross-box drift**: 1σ ≈ **0.039**
>
> 0.04 is 2σ of the *worst* (cross-box) noise. The whole 208–216 alibi+X batch
> landed at Δ 0.005–0.025 — all swallowed by the band, all NULL. If any was a real
> +0.01–0.02 stacking win, this screen **cannot see it** (this is exactly the
> regime modded-nanogpt / parameter-golf resolve, by pairing treatment vs control
> in the *same* session and averaging seeds, never letting drift in). **Fix:** judge
> each treatment paired against a same-session/same-box control + ≥3-seed median,
> which collapses drift → a real ~0.01–0.015 band. Not yet wired into `finalize_one`.

### Stage 2 — CONFIRM (3 seeds, band 0.02) — required before any promotion
Only a Stage-1 candidate WIN enters Stage 2. Run **both** the challenger and the
current champion at **3 seeds — `42, 123, 7`** (champion's 3-seed mean is
measured once when it is first crowned and cached, so in steady state only the
challenger's 2 extra seeds are new GPU).

```
PROMOTE  ⟺  mean3(challenger) < mean3(champion) − 0.02
```

- **0.02 ≈ 2·SEM** of a 3-seed mean (`σ/√3 ≈ 0.012`). Averaging 3 seeds shrinks
  the noise window by √3 ≈ 1.73× vs the single-seed 0.04 — a smaller window, not
  zero.
- **Like-for-like only.** Never compare a 3-seed challenger mean to a 1-seed
  champion val. Both sides are 3-seed means at the same tier/box.
- **Inside the band ⇒ do not promote.** `mean3(challenger) < mean3(champion)` but
  within 0.02 is "promising, inconclusive": keep the champion, log it, move on.
  (Optional stronger gate if you want extra safety on a marginal call: require
  all 3 challenger seeds individually below the champion mean — sign-consistency.)

On promotion: pin `mean3(challenger)` as the new champion val + its 3-seed std
(set `band = max(0.02, 2·std3)`), append to `champion.json.lineage`, and the new
champion becomes the Stage-1 bar for the next batch.

## What is NOT in scope (still hard rules)
- Still **one tier** — `tiny1m3m`. 3 seeds means 3 runs of the *same* tier, not a
  multi-tier ladder.
- Screening is still **strictly one seed**. No seed sweeps in Stage 1, no "add a
  seed to break a tie." A sub-noise Stage-1 effect is inconclusive, full stop.
- Three seeds is the **ceiling**, only for the promotion confirm. Not 5, not 10.

## Implementation hook (for whoever wires this into the daemon)
Today `finalize_one`/`promote_champion` in `queue-daemon.sh` promote directly off
the Stage-1 verdict. To add Stage 2:
1. On a Stage-1 candidate WIN, flip the idea to a new `needs-confirm` status
   instead of auto-promoting; enqueue a 3-seed confirm run (seeds `42,123,7` — a
   `run.json` `seeds` array, or 3 arq launches).
2. Ensure the champion has a cached 3-seed mean (`baseline.sh` gains a
   `measure3`/`pin3`); measure it once at crowning.
3. Promote only if `mean3(chal) < mean3(champ) − max(0.02, 2·std3)`; else flip
   `done` (NULL-confirmed) and log to `closed.md`.
This is intentionally **not built yet** — there is no winner to confirm. Build it
when the first Stage-1 candidate WIN appears.
