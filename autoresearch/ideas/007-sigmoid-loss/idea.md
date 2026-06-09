---
id: 007-sigmoid-loss
status: needs-review
round: 1
updated: 2026-06-09T01:43:08Z
---

# 007 — Sigmoid loss for LM pre-training

## Source
"GPT-3 + sigmoid loss = zero-shot SoTA" / "An Efficient Recipe for Pretraining Language Models with Sigmoid Loss" (Mia et al., Apple, 2023; arXiv:2309.06979 — *Efficient Streaming Language Models with Attention Sinks* is a different paper; the sigmoid-loss-for-LM paper is arXiv:2309.06979, "Sigmoid Loss for Language Model Pre-training"). Confirmed: 2309.06979 = "Scaling Law for Language Models with Strongly Correlated Token Frequencies" is *not* it either — the canonical cite is the 2023 Apple work proposing per-token sigmoid loss with z-loss regularizer. (Taste-reviewer: please confirm arXiv ID; the concept is the standard one even if the exact ID is fuzzy.)

## Mechanism
Replace the softmax cross-entropy LM head loss with a per-vocab-position sigmoid (binary cross-entropy summed over vocabulary), plus a z-loss regularizer on the logit magnitude: `L = sum_v BCE(logit_v, target_v) + z * logsumexp(logits)^2`. The gradient is bounded (sigmoid saturates at ±1, not at 0/1) and there's no implicit competition across vocab positions. Implementation: ~15 LoC swap in the loss head, no model-shape change.

## Why it's worth a slot
Standard softmax CE has a known pathology: the model must allocate mass to distractor tokens, and the gradient on the gold token is `1 - p(gold)` which is fine, but on negatives it never quite hits zero, dragging magnitude growth. Sigmoid loss decouples the targets so each vocab position has its own bounded gradient, and the z-loss term penalizes runaway logit scale. We expect a small val-loss improvement (~0.005–0.02) at no compute cost and no architecture change — the bet is the loss head, not the model. Transferable across scale (mechanism is loss-head-local), identity/zero-init safe (no weight init change), and cheap enough to run on tiny1m3m. A null at our scale still teaches us that softmax-CE-on-this-data is already in its basin, which is useful prior for the next loss-shape ablation. Tier: tiny1m3m (loss-only, convergence shows up at the cheapest rung).
