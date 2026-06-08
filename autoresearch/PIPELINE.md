# Idea pipeline

Multi-agent loop that takes an idea from "scouted" to "ran on hardware".
Agents are triggered manually and occasionally; each one **polls** the idea
folders, claims work by reading + flipping a status field, and stops when its
queue is empty. No agent talks to another directly — the only channel is the
`status` field in each `idea.md` frontmatter and the `review.md` log.

## The one source of truth: `idea.md` frontmatter

```yaml
---
id: 001-cautious-muon
status: needs-review
round: 1
updated: 2026-06-08T16:50
---
```

- `status` — **routing key**. Says *where in the pipe* the idea is, never a verdict.
- `round` — review-cycle counter. Caps the reviewer↔reviser loop (see below).
- `updated` — ISO timestamp, bumped on every status flip. Crash-recovery handle:
  an `-ing` status with a stale `updated` = a dead agent; reset it to its `needs-*`.

There is **no `owner` field and no separate state file** (`feedback.md` is dead).
Status alone routes. Verdicts live only in `review.md`.

## Status vocabulary

Queued (any matching agent may claim):

| status | claimed by |
|---|---|
| `needs-review` | reviewer |
| `needs-revision` | reviser |
| `needs-plan` | code-implementer |
| `needs-run` | run scheduler (human / Kaggle harness) |

In-flight (acts as the lock — one agent holds it):

`reviewing` · `revising` · `planning` · `running`

Terminal:

| status | meaning |
|---|---|
| `done` | ran, `evidence.md` written, win or null logged |
| `rejected` | killed in review; folder moved to `autoresearch/ideas/_closed/` |

## State machine

```text
scout/miner ─► needs-review
                   │  reviewer claims → reviewing → appends review.md r_n with a verdict
                   ▼
            ┌──────┴───────┬───────────────┐
         approve         revise          reject
            │               │               │
            ▼               ▼               ▼
        needs-plan    needs-revision     rejected ─► move to _closed/
            │               │
         planning      reviser claims → revising → edits idea.md, round++
            │               │
            ▼               └─────► needs-review   (re-review)
        needs-run
            │
         running ─► done   (evidence.md written)
```

## The claim protocol (every agent, every run)

1. `grep -l "status: <my-queue-state>" autoresearch/ideas/*/idea.md` — find my work.
2. For each hit: flip `status` to my `-ing` lock + bump `updated`. This claims it.
3. Do the work.
4. Flip `status` to the next queue state (+ `round++` for the reviser) + bump `updated`.
5. Repeat until no hits remain, then stop.

The reviewer's append-to-`review.md` and its status flip happen in the **same
pass** — never one without the other, or the log and the pointer desync.

## review.md format (append-only, newest round on top)

```markdown
# Review log — NNN <name>

## r2 — 2026-06-08 — verdict: approve
- ...

## r1 — 2026-06-08 — verdict: revise
- finding 1
- finding 2
```

Verdict is exactly one of `approve` / `revise` / `reject`. It sets `status`
(`approve→needs-plan`, `revise→needs-revision`, `reject→rejected`).

## Hard rules

- **3-round cap.** On `round: 3` the reviewer may only `approve` or `reject` —
  `revise` is forbidden. No idea cycles more than 3 times.
- **Cost-gate the loop.** The review loop exists to stop bad ideas *before they
  burn compute*. Only gate the expensive ones:
  - tiny1m3m ideas (~2 min on a T4): scout sets `status: needs-run` directly,
    skipping review.
  - screen20m+ ideas: full review loop (`needs-review`).
- **Rejects leave the scan path.** `rejected` → move the folder to
  `autoresearch/ideas/_closed/` and append a line to the CLOSED section of
  `autoresearch/queue.md`. Active greps stay clean.
- **One verdict per review pass.** A review that ends without exactly one
  verdict is malformed.

## Agent → prompt map

| Agent | Prompt | Greps | Writes |
|---|---|---|---|
| scout (in-repo) | `autoresearch/prompts/idea-scout.md` | — | `idea.md` (`needs-review` or `needs-run`) |
| miner (external) | `autoresearch/prompts/idea-miner.md` | — | `idea.md` (`needs-review` or `needs-run`) |
| reviewer | `autoresearch/prompts/idea-reviewer.md` | `needs-review` | appends `review.md`, flips status |
| reviser | `autoresearch/prompts/idea-reviser.md` | `needs-revision` | edits `idea.md`, `round++` |
| code-implementer | `autoresearch/prompts/code-implementer.md` | `needs-plan` | `plan.md` + code, → `needs-run` |
