---
id: 109-kda-channel-gate
status: needs-taste
round: 1
updated: 2026-06-13T05:32:00Z
transfer-risk: low
plain: It tries to let each memory channel forget at its own pace instead of forcing one shared decay for the whole head.
---

# 109 — KDA Channel Gate

## Source
Kimi Linear: An Expressive, Efficient Attention Architecture, arXiv:2510.26692 https://arxiv.org/abs/2510.26692; MoonshotAI/Kimi-Linear https://github.com/MoonshotAI/Kimi-Linear

## Mechanism
Replace the single decay/forget gate in a delta-rule or linear-attention block with a per-channel diagonal gate. Initialize the added gate as a no-op so the module starts as the baseline recurrence and only learns finer memory control if it helps.

## Scale evidence
The report trains Kimi Linear at 3B activated / 48B total parameters and evaluates on 1.4T tokens, with consistent gains over full-attention baselines, so transfer risk is low.

## Why it's worth a slot
This isolates the one part of Kimi Linear that looks most like a compact mechanistic lever: finer-grained memory decay; a null says the hybrid recipe matters more than the gate itself.
