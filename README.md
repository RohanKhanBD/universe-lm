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

Open the branch — its README has the exact commands and a copy-paste prompt for
your AI coding agent.

| Experiment | What it tests | Smallest run | Branch |
|---|---|---|---|
| **Attention Residuals (AttnRes)** | Replace the fixed inter-layer residual with softmax attention over depth ([arXiv:2603.15031](https://arxiv.org/abs/2603.15031)). A depth lever — should help more as layers grow. | ~1 hr (8M rung) | [`experiment/attn-res-v1`](https://github.com/vukrosic/universe-lm/tree/experiment/attn-res-v1) |

*Want to add your own? Branch from `main`, wire the idea behind a config flag,
add a branch README with the run commands, and open a PR adding a row here.*

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
