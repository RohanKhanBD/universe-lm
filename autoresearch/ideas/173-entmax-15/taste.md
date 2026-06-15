## r1 — 2026-06-15 — verdict: revise

**Leverage is soft.** Paper gain is modest (+0.5 BLEU on WMT'14, +0.5–1.0 GLUE on BERT-base at ≥100M) and the miner's own predicted band Δval ∈ [-0.005, -0.020] has the lower end sitting inside the |Δ|<0.01 null band — 60% of the expected range is null. Not a high-leverage lever at our tier.

**The bet is a vibe.** "Sparse attention is a strong inductive bias that may shorten the optimization horizon" is hand-wavy. A revise needs ONE sharp mechanistic sentence — e.g., "at H=4 d_k=16 T=2048, the dense AV matmul accumulates gradient noise on the bottom ~70% of K positions; forcing α_h>1.5 collapses that mass into the top ~30%, so per-step gradient SNR on the surviving K rows rises by ~3×." Predict something concrete.

**Softmax-replacement axis is weakly-answered here.** Three siblings have already nulled in this family:
- 148-focal-mod null — "non-softmax-attention axis at 0.94M" (closed)
- 156-moa null — parallel-attention-experts + router (closed)
- 166-t5-rpe / 152 / 155 / 160 / 162 / 165 — every per-head attention-shape lever null at 0.94M/12L/4H
- closed.md axes line: "NSA / diff-attn / hybrid heads" closed

Diff-attn is the closest cousin (smooth differentiable post-QK operator) and is explicitly on the closed axis. A null from entmax-1.5 would re-confirm what 148 + 156 + 152 + 155 + 160 + 162 + 165 already weakly-answered — the softmax-replacement / per-head-shape family does not bind at 0.94M. The miner acknowledges 148 but argues entmax-1.5 is a different operator (true sparsity, not gated-additive context, and not focal modulation). That distinction is real but the queue has 9 prior soft signals in the family.

**Field-veto signal.** 7 years after Peters et al. (2019), LLaMA 1/2/3, Mistral, Qwen 1/2/2.5, Gemma 1/2, OLMo, Falcon all use softmax. The mechanism has had 6+ years to be adopted; it wasn't. That is not proof it's bad, but it IS soft evidence the lever is in the right tier, not ours.

**Transfer case is plausible but unvalidated for causal LM.** Miner's "transfer-risk: med" is honest. Closest 135M re-evaluation would be in Phase-2; the lever needs to clear tiny1m3m to get there.

**Engineering is good.** Bit-identical step-0 init via `α_h = 1 + 0.5·(1+tanh(α_raw_h))` is correct (the three-way init comparison is well done), bisection budget is realistic, LoC ~80 is fine, helper is well-known. None of this is the problem.

### What the r2 pitch must add
1. **One sharp mechanistic sentence** that names the binding constraint at 0.94M/12L/4H/3M tokens and predicts which axis entmax-1.5 will move. Vibe: not enough.
2. **A stronger expected Δ** — at least Δ ≤ -0.015 committed (the bar —current best 154-rebased-attn WIN was Δ=-3.48 against a buggy ctrl, plan bar is -0.005/-0.01). If the miner's honest read is "this is a null-confirmation play", say so explicitly with a stated info-value.
3. **Explicit differentiation from 148/156/152/155/160/162/165/166**: argue WHY entmax-1.5 (true sparsity, operator replacement) is a different family than the eight prior soft siblings. The "we are softmax at step 0" framing is the strongest differentiation; pin it down.
4. **Optional**: consider a milder-sparsity variant (entmax-1.2 — close to softmax, only mildly sparse) as a "small dose" of the lever; if the r2 pitch goes that way, the bet becomes "a small dose of sparsity helps" and the Δ floor can be -0.005 with confidence.
