# Generate-ideas prompt (idea generation only — no running, no gates)

Search for new research levers and file them as idea files. **That's the whole
job.** Do NOT plan, implement, or run anything. Do NOT touch the pipeline WIP
gate, the re-pitch queue, or `queue.md`. Just find ideas and write them down.

---

> ## 🔴 YOU RUN UNATTENDED — ACT, DON'T ASK
> This fires from a button with no human watching. File the ideas and stop.
> Never end by asking "want me to mine more?" — the answer is always yes, so
> just do the full batch this pass, then stop.

> ## 🔴 THE FIXED TEST — context for what counts as an idea
> Every idea in this repo is eventually run at **tiny1m3m (0.94M params · 3M
> tokens), one seed: 42.** So only file levers that are **mechanisms**
> (architecture / optimizer / loss / positional encoding), implementable in
> **< 200 lines** behind a config flag, and **identity/zero-init** (step-0 ≈
> baseline). No hyperparameter tuning, no tokenizer/data changes, no
> inference-time tricks.

**Repo:** `/Users/vukrosic/my-life/llm-research-kit-scaling`

**Generate exactly 3 new ideas this pass.**

## Step 1 — Dedup

Read `autoresearch/closed.md` (the dedup list) and skim
`ls autoresearch/ideas/` so you don't re-file anything that already exists.

## Step 2 — Search

Use the `mcp__bing-search__bing_web_search` tool plus these sources:
- arXiv `cs.LG` / `cs.CL` — keywords: Muon, orthogonal, spectral, RoPE, relative
  position, linear attention, DeltaNet, normalization, softmax alternatives,
  cautious, MoE routing.
- kexue.fm (Su Jianlin) — dense source of mechanism ideas; read Chinese natively.
- X: @kellerjordan0, @borisdayma, @hardmaru, @cloneofsimo, @BlinkDL_AI.
- Repos: modded-nanogpt, llm.c, fla-org, state-spaces, Dao-AILab.

Prefer levers with **published gains at ≥100M scale** (lower transfer risk).

## Step 3 — File each idea

For each of the 3 ideas, find the next free 3-digit number
(`ls autoresearch/ideas/`), then write
`autoresearch/ideas/NNN-<slug>/idea.md` (slug = kebab-case, 1-3 words):

```markdown
---
id: NNN-<slug>
status: needs-taste
round: 1
updated: <ISO timestamp>
transfer-risk: <low|med|high>
plain: <ONE line, zero jargon — what this idea tries, explained to a non-expert>
---

# NNN — <Title Case Name>

## Source
<paper title + arXiv id / repo / post URL. Prefer 2025-2026 work.>

## Mechanism
<1-2 sentences: the math or operation. Must be < 200 LoC in this repo.>

## Scale evidence
<largest scale the source showed gains at, and a one-line justification for the
transfer-risk tag: low (≥100M), med (10-100M or strong argument), high (toy only).>

## Why it's worth a slot
<the bet in one sharp sentence: we expect X because Y; a null still tells us Z.>
```

The `plain:` line is **required** — one sentence, no jargon, so anyone can read
the idea list and understand what it's trying.

## Step 4 — Log and stop

Print a short log: for each filed idea, its `NNN — name`, source, and one-line
plain summary. Then stop — don't plan, implement, or run anything.
