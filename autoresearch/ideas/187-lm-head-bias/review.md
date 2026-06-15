## r1 — 2026-06-15 — verdict: revise

- **vocab_size is wrong (6× off) → param-count claim breaks.** idea.md:45 says
  `vocab_size = 8192 (typical for tiny1m3m; verify from the config)` and uses
  that to compute `8192 params (+0.87% of 0.94M)`. The actual `LLMConfig.vocab_size`
  is **49152** (configs/llm_config.py). True overhead is **49,152 params = ~+5.2%
  of 0.94M** — a six-fold correction and a *sizeable* param injection, not the
  "0.87%" figure. This changes the portfolio-fit argument (a +5% lever is not
  "small but not negligible" — it's larger than the entire budget for a typical
  OutputHead batch lever). The plan.md (OH5 VocabBias) explicitly tags it
  "many params but trivial compute"; please update the intution + portfolio
  framing to match.
- **baseline-cache reference is stale.** idea.md:58 cites
  `autoresearch/baseline-cache.json box 5b8a7fea8963 (RTX 3060), val_mean = 6.3988,
  noise_band = 0.04, n_measurements = 3`. The current pinned cache
  (autoresearch/baseline-cache.json, measured 2026-06-15T07:04:48Z) shows
  **`val_mean = 6.2403, n_measurements = 3` for that box_key**. The 6.3988
  number is the *pre-175-alibi* baseline; 175 alibi-slopes is a closed WIN
  (Δ-0.1585) and the champion stack is now `Tiny1M3MAlibiConfig` (per
  configs/llm_config.py, which is what `Tiny1M3MLogitScaleConfig` etc. subclass).
  The pass/fail bar must be against the **current champion**, not the pre-ALiBi
  baseline. Either (a) restate the bar as `Tiny1M3MAlibiConfig + use_vocab_bias`
  vs `Tiny1M3MAlibiConfig` baseline, or (b) keep the cache reference but pull a
  fresh number on run day and clearly mark this as the ABLATIVE Δ (i.e.
  vocab-bias-only vs the champ stack).
- **Lever is already implemented as OH5 VocabBias — say so in the plan, don't re-derive.**
  The exact lever `logits += b_v` with `b_v = zeros(vocab_size)` is already wired:
  - config flag `use_vocab_bias: bool = False` (configs/llm_config.py, line ~N
    in the OH5 VocabBias comment block)
  - parameter allocation `self.vocab_bias = nn.Parameter(torch.zeros(config.vocab_size))`
    gated on `use_vocab_bias` (models/llm.py)
  - forward hook `if self.use_vocab_bias: logits = logits + self.vocab_bias`
    after softcap (models/llm.py)
  The plan.md (docs/research/output_head/plan.md, Batch 2 row OH5) is this
  exact mechanism. The idea.md source/mechanism section is a clean re-statement
  but doesn't acknowledge either the config flag or the plan.md entry. The
  reviser must: (a) add an `## Existing wiring` section naming `use_vocab_bias`
  + the line in models/llm.py, (b) update the Design sketch to add a
  `Tiny1M3MAlibiLMHeadBiasConfig(Tiny1M3MAlibiConfig)` subclass (stacking on
  the current champion, matching the 184-logit-scale precedent) rather than a
  raw `Tiny1M3MConfig` subclass, (c) drop the redundant mechanism prose and
  instead cite the OH5 row in plan.md as the canonical spec.
- **Defensive comparison needs to cite OH5/plan.md, not just closed.md.** The
  idea.md "Distinct from closed axes" section claims "no prior lever in the
  repo tests a per-vocab LM head bias." That statement is true vs `closed.md`
  but false vs `docs/research/output_head/plan.md` (OH5 row). The plan row
  frames the lever as "mostly re-learns token frequency, a known small CE
  win"; the 187 pitch frames it as "output/input decoupling via per-vocab
  scalar." Both framings are correct, but the planner needs to pick one (or
  unify them) so the run spec is unambiguous about the *expected mechanism*
  vs the *expected magnitude*. Pre-binding on "expected Δ ~ -0.005 to -0.02"
  (OH5's framing) vs "expected Δ bound by the per-token decoupling capacity
  at 0.94M" (187's framing) leads to different pass bars.
- **Step-0 byte-identity claim is sound, but verify against champion.** The
  `lm_head_bias = zeros(V)` ⇒ `logits + 0 = logits` argument is correct for
  any tied-or-untied head. The implementer must verify
  `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0` AND
  `max_abs_diff(trt_step0_loss, ctrl_step0_loss) == 0.0` where the ctrl is the
  **champion stack (Tiny1M3MAlibiConfig)**, not plain Tiny1M3MConfig.
- **Transfer-risk: low holds.** T5 220M-11B validation is genuine; the
  frontier-decoder abandonment is a capacity argument not a mechanism
  invalidation. Fine as-is.
- **Falsifiable bar exists, but the threshold needs re-anchoring.** Current bar
  is `trt_val ≤ ctrl_val_mean - 0.005`. With `ctrl_val_mean` set to the
  pre-ALiBi 6.3988 baseline, that bar is *too loose* (we already know 175
  cleared 6.24). With it set to 6.2403 + the expected OH5 lever magnitude
  (-0.005..-0.02), the bar becomes meaningful: a vocab-bias win on top of ALiBi
  would have to clear `≤ 6.2353` (champion - 0.005) to count, and the noise
  band is ±0.04. Tighten or accept borderline-null, but make the threshold
  match the *actual* champion stack you're stacking on.

**Next step:** Reviser — apply the four corrections above (vocab_size
49152, baseline-cache fresh pull, existing-wiring acknowledgement, subclass
on Tiny1M3MAlibiConfig). Then re-flip to `needs-review` for another pass.
