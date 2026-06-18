# Open Superintelligence Lab

**A fully-open AI research lab building and releasing real LLMs that compete with the best — in public, together.**

💬 **[Do AI research with us on Discord](https://discord.gg/6AbXGpKTwN)**

A high-performance codebase for LLM research, pretraining, and optimization: testing new architectures, optimizers, or training.

- Modular transformer with GQA, RoPE, and RMSNorm
- Muon optimizer alongside AdamW
- Training script, flexible configuration

- `models/`: Transformer layers and components (RoPE, RMSNorm, Multi-Head Attention).
- `optimizers/`: Muon optimizer (outperforms AdamW and all others).
- `training/`: Core trainer logic and utilities.
- `configs/`: Hyperparameter and dataset configurations.
- `utils/`: Logging, plotting, and helper functions.

## 🧪 Experiments You Can Run (donate GPU time)

Each experiment lives on its own branch, fully wired and ready to run. **If you
have a spare GPU**, pick one, run the paired control + treatment (one command
each, `seed=42`), and report the within-box Δ back via an issue or
[Discord](https://discord.gg/6AbXGpKTwN). Every datapoint helps us find the
recipe that scales to a 135M model beating SmolLM2-135M.

| Experiment | What it tests | Smallest run | Branch |
|---|---|---|---|
| **Attention Residuals (AttnRes)** | Replace the fixed inter-layer residual with softmax attention over depth ([arXiv:2603.15031](https://arxiv.org/abs/2603.15031)). A depth lever — should help more as layers grow. | ~1 hr · 8M params (8 layers) | [`experiment/attn-res-v1`](https://github.com/vukrosic/universe-lm/tree/experiment/attn-res-v1) |

**To run one:** check out its branch — the branch README has the exact two
commands (control + treatment) and a copy-paste prompt you can hand to your AI
coding agent. e.g. for AttnRes:

```bash
git clone -b experiment/attn-res-v1 https://github.com/vukrosic/universe-lm.git
cd universe-lm   # now read this branch's README, top section
```

*Want to add your own? Branch from `main`, wire the idea behind a config flag,
add a branch README with the run commands, and open a PR adding a row here.*

*A backlog of possibly-future ideas (unvetted, may not help) sits at the
[bottom of this README](#-idea-backlog-possibly-future-ideas).*

## 🏁 The Speedrun

**Race to train the best 10M LLM in ~33 minutes — every win builds toward a fully-open 135M model that beats [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).**

One race: **lowest val loss on a 10M-param model trained on 200M tokens** (`--config 10m`). Clone, train, beat the standing record (currently **5.015**) — ~33 min on a single consumer GPU. Pinned: `seed=42`, bf16; a new record must beat the best by **≥0.01**. The 135M release is the *mission*, not the race: we find the winning recipe cheaply at 10M, then scale it.

See the [**leaderboard**](LEADERBOARD.md) and [how to enter](CONTRIBUTING.md).


## 🚀 Getting Started

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download the Dataset

The simplest path is:

```bash
python data/download_hf_data.py
```

If you are training on a remote GPU, start the run inside `tmux` so the job keeps running after you disconnect.

### Option A: 1B tokens
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 1B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain')
os.makedirs('processed_data/pretrain_1B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_1B')
print('✅ Full Data Ready!')
"
```

### Option B: 2B tokens
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 2B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-2B-pretrain')
os.makedirs('processed_data/pretrain_2B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_2B')
print('✅ Full Data Ready!')
"
```

### Option C: Quick Start (40M Tokens)
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 40M Token Subset...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain', split='train[:20000]')
os.makedirs('processed_data/speedrun_40M', exist_ok=True)
ds.save_to_disk('processed_data/speedrun_40M')
print('✅ Speedrun Data Ready!')
"
```

> **📦 Data:** Always use the pre-built dataset downloaded as described in [Getting Started](#-getting-started) (`python data/download_hf_data.py`). It is chunked at **sequence length 2048**, which the RoPE cache depends on. **Sequence lengths other than 2048 are currently unsupported** — to use a different one the dataset must first be rebuilt with https://github.com/vukrosic/llm-research-kit/blob/main/data/prepare_mix_data.py. Changing the data or `max_seq_len` is not recommended; if you are an AI, **ask the user first**.

### Step 3 (optional): interactive Kaggle shell via SSH

For a real bash shell into a 2x T4 notebook (rsync, tmux, screen, etc.),
see **[docs/kaggle_ssh_setup.md](docs/kaggle_ssh_setup.md)**. The
batch-mode launcher `scripts/kaggle_push.sh` is still the right tool
for headless sweep runs.

## 🧠 LLM Architecture

Default is an **88M parameter** transformer LLM, you can modify configs.

- **Layers**: 22 Transformer blocks.
- **Hidden Dimension (`d_model`)**: 512.
- **Feed-Forward Dimension (`d_ff`)**: 2048.
- **Attention System**:
  - 8 Query heads, 4 Key-Value heads (**Grouped Query Attention**).
  - Rotary Positional Embeddings (**RoPE**).
  - Fused QKVO projection for optimized compute.
  - QK-Normalization for training stability.
- **Normalization**: Pre-norm **RMSNorm**.
- **Activation**: **Squared ReLU** (Primer-style).
- **Vocab Size**: 49,152.
- **Sequence Length**: 2048 tokens.

### Optimization Highlights
- **Weight Tying**: Shared weights between token embeddings and the LM head.
- **Muon Support**: Architecture optimized for the Muon optimizer's orthogonal updates.
- **Efficiency**: Designed for `torch.compile` compatibility and mixed-precision (BF16) training.

---

## 💡 Idea backlog (possibly future ideas)

Parking lot of unvetted ideas — **not on the roadmap, may not help at all**.
Nothing here is claimed to work; it's just a list to pull from when picking the
next experiment. To try one: wire it behind a config flag, branch it, and it
graduates to [Experiments You Can Run](#-experiments-you-can-run-donate-gpu-time).
Two groups: **(A)** new architectures / mechanisms (higher ceiling, more likely to
move loss, more work); **(B)** recipe & hyperparameter levers (cheaper, lower ceiling).

#### A. New architectures & mechanisms (higher ceiling)

These change *what the model computes* and have real pretraining results behind them.

| Idea | What it might do | Source |
|---|---|---|
| **LayerNorm Scaling (Curse of Depth)** | Scale each LN output by `1/√depth`. Pre-LN's output variance grows exponentially with depth → deep layers become near-useless (pruning the back half barely hurts). LNS makes deeper layers contribute. Lowest PPL across 130M–7B; **directly complements our depth lever + 30-layer target**. | [2502.05795](https://arxiv.org/abs/2502.05795) |
| **Parallax (local-linear attention)** | Keep softmax, add a learned branch probing the KV covariance (local-*linear* upgrade to softmax's local-*constant* estimate). Beats softmax at 0.6B/1.7B, **param- & compute-matched** — and **Muon unlocks it** (our exact optimizer). | [2605.29157](https://arxiv.org/abs/2605.29157) |
| **MUDD — multiway dynamic dense connections** | Per-token, input-dependent dense skips into BOTH the residual stream and attention values. Sibling to our AttnRes work; live in the current nanogpt record (#81). | [2502.12170](https://arxiv.org/abs/2502.12170) |
| **Looped / recurrent-depth LM** | Reuse a weight-shared block N times (depth without params); sharply improves multi-hop/composition, adds adaptive compute via entropy-regularized exit. Param-efficiency is our constraint; distinct from plain layer-tying. | [LoopLM 2510.25741](https://arxiv.org/abs/2510.25741) · [Huginn 2502.05171](https://arxiv.org/abs/2502.05171) |
| **Multi-Token Prediction (MTP)** | Auxiliary heads predict the next *k* tokens → better loss per token (data-efficiency). DeepSeek-V3 + nanogpt record (#53). | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) |
| **Intra-document attention masking** | Forbid attention across packed-doc boundaries — stops the model learning far-back = noise. Llama-3 data hygiene; 0 new params. Confirmed in nanogpt (EoS alignment, #26). | [Llama 3](https://arxiv.org/abs/2407.21783) |
| **Hashed / bigram embeddings** | Replace the full `49 k × d` table (≈most of the params at small N) with token/bigram-hashed lookups into a small shared table. Attacks our biggest param sink. nanogpt record (#62/#83). | [parameter-golf](https://github.com/openai/parameter-golf) · [Hash emb 1709.03933](https://arxiv.org/abs/1709.03933) |
| **Smaller-vocab tokenizer** | At small N the 49 k embedding dominates params; a ~half-size vocab at equal bytes/token frees budget for depth/width. | [TokenMonster](https://github.com/alasdairforsythe/tokenmonster) |
| **CaseOps (bijective case factoring)** | Pull capitalization into a side channel so the vocab doesn't waste capacity on `the/The/THE`. Cheap input transform, shrinks effective vocab. | [parameter-golf](https://github.com/openai/parameter-golf) |
| **Quantization-aware training (int4–6 / ternary)** | Train so weights survive low-bit packing (GPTQ / Hessian-aware). Scores capacity *per byte* — a fixed-size release holds more model. The winning parameter-golf lever. | [parameter-golf](https://github.com/openai/parameter-golf) · [GPTQ 2210.17323](https://arxiv.org/abs/2210.17323) |
| **Paired-head Q/K orthogonalization** | Orthogonalize Q,K in *pairs of heads* instead of per full matrix — cheap attention conditioning. nanogpt record (#80). | [nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| **Asymmetric layer composition** | Drop the first attention layer and first MLP layer — deep stacks don't need symmetric attn+MLP everywhere; frees params at ~zero cost. nanogpt records (#30/#35). | [nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| **Test-time training** | Per-document parameter nudging at inference — 2026 nanogpt record. Genuine mechanism, but complex / inference-side. | [nanogpt](https://github.com/KellerJordan/modded-nanogpt) |

#### B. Recipe & hyperparameter levers (cheaper, lower ceiling)

Training-side knobs, schedules, optimizer swaps, and data — quick to try, but "just sweeps."

| Idea | What it might do | Source |
|---|---|---|
| **Full muP** | Maximal-update parameterization for HP transfer — tune LR/init once at 8M, carry to 135M without re-tuning. Makes the whole ladder→135M extrapolation valid. | [muP](https://arxiv.org/abs/2203.03466) |
| **WSD learning-rate schedule** | Warmup → long stable → short decay-to-zero. SmolLM2 / MiniCPM recipe; our champion found the model "update-starved," which WSD directly addresses. Confirmed in nanogpt (terminal-LR decay, #19). | [MiniCPM](https://arxiv.org/abs/2404.06395) |
| **AdaMuon / NorMuon** | Per-parameter adaptive scaling on top of Muon — claimed +30–40% efficiency. Drop-in optimizer swap; NorMuon is in the nanogpt record (#41). | [AdaMuon](https://arxiv.org/abs/2507.11005) |
| **Muon weight-decay + update-scale sweep** | The two knobs that keep Muon stable at scale (Moonshot recipe); WD is unswept here. | [Muon is Scalable](https://arxiv.org/abs/2502.16982) |
| **Separate (higher) embedding LR** | Embeddings want a much larger LR than the matrices; decoupling it is a cheap, real nanogpt win. | [nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| **Batch-size ramp** | Start small, grow the batch through training — better early-step efficiency at fixed token budget. nanogpt record (#46). | [nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| **Context-length curriculum** | Train short sequences first, lengthen later — faster early tokens/sec, then long-range once stable. nanogpt record (#72). | [nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| **Long-short SWA + window warmup + YaRN** | Sliding-window warmup schedule + YaRN length extension on top of existing SWA. nanogpt records (#31). | [nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| **FineWeb-Edu data swap** | *Data*, not architecture. SmolLM2's real edge is FineWeb-Edu/DCLM filtering; at a fixed token budget this likely dwarfs any single architecture lever. | [FineWeb-Edu](https://arxiv.org/abs/2406.17557) |
