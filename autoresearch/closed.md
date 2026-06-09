# Closed levers — do not re-propose

The dedup list for the autoresearch loop. The **miner/scout read this before
filing** an idea; the **reviewer reads it** to catch closed re-proposals and
**appends to it on every `reject`**. One line per closed lever, terse and
greppable. (Full results live in `LEADERBOARD.md` — not edited by this system.)

## Who may append here

| Closer | When | Entry format |
|---|---|---|
| reviewer | verdict = `reject` (killed on paper) | `<NNN-slug or lever> — reject: <reason> — <date>` |
| evidence/run step | post-run null or failed pass-bar (killed by data) | `<NNN-slug> — null: <val, Δ vs ctrl> — <date>` |

The code-implementer never closes — if blocked it bounces the idea back to
`needs-review`. Keeping a single agent (reviewer) on the write path avoids races.

## Closed axes (seed — migrated from queue.md, 2026-06-08)

- V/Q/K/O embeds + combos, q_gain / k_gain (screen20m rows 0-17)
- SWA window sweep (256/384/512/768/1024/2048) — 512 winner
- RoPE base sweep — 500k winner
- NoPE, post-norm, layer tying, MHA vs GQA, MLA, Tied QK (on best baseline), dilated attention, logit softcap
- Norm zoo (pnorm, manhattan, center, squash, clip, channelscale)
- NSA / diff-attn / hybrid heads
- Multiscale heads / parallel block / attn sink (2026-06-04 batch)

## Closed by the loop (append below, newest first)

<!-- reviewer/evidence step appends one line per close here -->
- 002-cautious-adamw — null: A_emb=+0.0003 B_gain=-0.0066 at tiny1m3m (in noise; plan calls for screen20m tier, off-policy per user) — 2026-06-09
