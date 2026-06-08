# Idea queue (index)

**Each idea lives in its own folder under `autoresearch/ideas/<NNN-slug>/idea.md`.**
This file is just the index / status board / active-slot tracker. To add a
new idea: create a folder, write `idea.md`, append to the table below.

**Pipeline:** the review→revise→implement→run loop and the canonical `status`
vocabulary are defined in [`PIPELINE.md`](PIPELINE.md). The routing
truth is the `status:` frontmatter in each `idea.md`; the tables here are a
human-readable view of it. Regenerate with
`grep -H "status:" autoresearch/ideas/*/idea.md`.

## Active remote queue (FIFO, always 3)

| Slot | Folder | Run | Status |
|---|---|---|---|
| 1 | `autoresearch/ideas/001-cautious-muon/` | tiny1m3m + `use_cautious_muon=True` | IN-PROGRESS (Kaggle T4, s42) |
| 2 | empty | fill from PENDING below | — |
| 3 | empty | fill from PENDING below | — |

Rule: aim for 3 in the queue at all times so the remote never idles.

## Ideas board (in folder-number order)

`status` mirrors each `idea.md` frontmatter — see [`PIPELINE.md`](PIPELINE.md).

| # | Folder | One-liner | Expected Δ | status |
|---|---|---|---|---|
| 001 | `001-cautious-muon/` | sign-mask on Muon ortho'd update | −0.01 to −0.05 | running |
| 002 | `002-cautious-adamw/` | sign-mask on AdamW (1D params) | −0.005 to −0.02 | needs-revision |
| 003 | `003-polar-express/` | adaptive NS coeffs vs fixed polar_express | −0.01 to −0.03 | needs-revision |
| 004 | `004-moe-ffn/` | top-2 routing, 8 experts, aux load-balancing | −0.05 to −0.12 | needs-revision |
| 005 | `005-soap/` | Shampoo + Adam hybrid in eigenbasis | −0.02 to −0.05 | needs-review |
| 006 | `006-retnet-retention/` | linear-attention retention, parallel/recurrent modes | −0.02 to −0.06 (if transfers to 10M) | needs-review |

## PENDING — not yet foldered (migrate on first touch)

These are tracked here as a quick-lookup; copy to a numbered folder when
the idea is about to be run. Full spec lives in the table at the end of
this file until then.

Optimizer: Moonlight per-param RMS scaling · Decoupled Q/K from V in Muon routing · Cautious Lion · Schedule-free AdamW · EMA-of-orthogonalized-matrix.
Architecture: Gated DeltaNet · Soft MoE (fallback) · BigBird sparse · Sandwich block · Product-key FFN.
Loss/objective: Sigmoid loss / ET loss · PolyLoss.
Positional: FIRE · CoPE.

## CLOSED ideas (do not re-propose)

Moved to [`closed.md`](closed.md) — the loop's dedup list (miner reads it,
reviewer appends on `reject`). `LEADERBOARD.md` remains the full human results
record.

## External sources to mine (refresh weekly)

- arXiv: `cs.LG`, `cs.CL` — filter "Muon", "orthogonal", "spectral", "MoE", "state space", "Mamba", "DeltaNet", "linear attention", "cautious"
- X follows: @kellerjordan0, @borisdayma, @arankomatsuzaki, @_akhaliq, @hardmaru, @StasBekman, @cloneofsimo
- Repos: modded-nanogpt, picoGPT, llm.c, nanogpt-speedrun, PaLM-pytorch, mamba
- HF papers: https://huggingface.co/papers
- Papers With Code: https://paperswithcode.com/task/language-modelling

## Remote run log

| Date | Slot | Folder | Run | Status | Val loss | Δ vs ctrl |
|---|---|---|---|---|---|---|
| 2026-06-08 | — | — | tiny1m3m ctrl (1B data, T4) | DONE | 6.4287 | — |
| 2026-06-08 | 1 | `001-cautious-muon/` | tiny1m3m + cautious-muon, s42 | IN-PROGRESS | ? | ? |

## Protocol (per-idea folder)

When an idea moves to implementation, add these files alongside `idea.md`:

| File | When | Contents |
|---|---|---|
| `plan.md` | promoting to implementation | implementation spec, flags, controls, expected cost |
| `review.md` | parallel-AI code review | critique, suggestions, sign-off |
| `evidence.md` | after a run | val loss, commit link, Δ vs ctrl, verdict |
| `notes.md` | anytime | scratchpad, dead-ends, follow-up ideas |

To re-prioritize the queue: renumber folders. To close: set the `idea.md`
frontmatter `status` to `rejected` (killed in review) or leave at `done` (ran),
move a `rejected` folder to `autoresearch/ideas/_closed/`, and append a line to the
CLOSED section below. See [`PIPELINE.md`](PIPELINE.md) for the full
state machine.
