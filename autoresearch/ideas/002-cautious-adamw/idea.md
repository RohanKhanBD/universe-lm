---
id: 002-cautious-adamw
status: needs-revision
round: 2
updated: 2026-06-08T18:15
---

# 002 — Cautious AdamW

## Source
Liang et al. 2024, "Cautious Optimizers" (arXiv 2411.16085). Extension of [[001-cautious-muon]] to the AdamW path (1D params: gains, scalars, embeddings, head).

## Mechanism
Same sign-mask as Cautious Muon, applied to the AdamW update for 1D parameters. The mechanistic claim from the paper is that the mask helps when the *preconditioned* update direction disagrees with the current gradient sign — that disagreement is the stale-momentum / 2nd-moment-scaling artifact. On Muon this is common (orthogonalized update is sign-agnostic by construction); on AdamW it is rarer in steady state because 2nd-moment normalization already pulls the update toward the sign of the gradient, so the mask is mostly a no-op. The gain on AdamW is therefore expected to be smaller than on Muon (Liang et al. Table 1), but the *failure mode* it targets is different — and complementary, not redundant. A null on this idea does NOT imply Muon-cautious was useless; the two paths are independent and the paper reports both as additive in their small-scale ablations.

## 1D params, split
The 1D bucket covers (a) RMSNorm gain, (b) any learned scalar, (c) embedding rows, (d) LM head. The mask is *very* meaningful for embedding rows (a wrong sign on a rarely-used row's update is worse than a zero update) and *barely* meaningful for a constant-sign gain. A null on the full 1D bucket is uninformative — a split A/B is the only way to read the result:
- **condition A:** mask on embedding rows only
- **condition B:** mask on gains + scalars only
- **condition C (optional):** mask on all 1D (current proposal)

A null on A and a hit on B → keep B, drop A. A null on B and a hit on A → keep A, drop B. Both null → close the idea. Both hit → keep both.

## Wiring
`use_cautious_adamw` is **not** a config field. `configs/llm_config.py:358-360` mentions it as a future flag but the AdamW 1D path is computed inside `Muon.step()` (`optimizers/muon.py:147`, the `adamw_lr` group), not in a separate `AdamW` class. **Decision: extend `Muon` to drive the AdamW path's mask.** Add a top-level `use_cautious_adamw: bool = False` to `LLMConfig`, thread it into the `Muon` optimizer group as `cautious_adamw`, and apply the same `(update * (update.sign() == grad.sign())).masked_fill(~, 0.0)` step in the 1D branch of `step()`. ~5 LoC, bit-identical to baseline when False.

## Run notes
- Run only after [[001-cautious-muon]] passes Phase 1 (tiny1m3m val ≤ 6.4206). If 001 fails, close this idea too — the same mechanism didn't generalize.
- **Tier:** screen20m is the only tier where this is resolvable. tiny1m3m at ~8M training tokens has noise ±0.06-0.16 (`LEADERBOARD.md` line 96-99); the expected Δ is below the noise floor there.
- **Expected Δ:** `−0.005 to −0.01` on screen20m (per-parameter, not per-seed mean), with `−0.02` as a stretch outcome. A null is informative, not a failure.
- **Seeds:** ≥3 seeds (42/43/44) for the screen20m run; otherwise a sub-1% effect closes on noise.
- **Control:** the current screen20m best baseline (V+q+SWA+HighRoPE 4.6364, `LEADERBOARD.md` row 18d) — same control as [[001-cautious-muon]]'s screen20m follow-up so the two A/Bs are directly comparable.
- **Fallback:** if running 001's screen20m follow-up lands first, reconsider whether a standalone 002 is still worth a fresh run — the AdamW-1D-mask hypothesis can be re-litigated with 001's screen20m numbers as a second contrast at zero extra cost.

(Pipeline status lives in the frontmatter above.)
