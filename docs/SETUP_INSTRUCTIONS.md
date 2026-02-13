# 🚀 LLM Research Kit: Setup Guide

Welcome to the **LLM Research Kit**! This repository is designed for high-performance LLM research, providing a clean and efficient environment for pretraining and optimization experiments.

---

## Step-by-Step Instructions

### Step 1: Prepare Your Environment

We recommend using **Python 3.10+**.

#### Clone the Repository
```bash
git clone https://github.com/vukrosic/llm-research-kit
cd llm-research-kit
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download the Dataset

You can download a small subset of data for quick testing or the full dataset for large-scale training.

#### Small Scale (Recommended for initial tests)
```bash
python3 data/download_hf_data.py
```

### Step 3: Run Your First Training

To verify your setup, run the base training script:
```bash
python train_llm.py
```

This will train a small model on a default number of tokens. You can monitor the progress and final validation loss in the terminal.

---

## 🛠️ Research & Iteration

### Configuration
Modify `configs/llm_config.py` to adjust model architecture, learning rates, and optimization schedules.

### Model Architecture
Edit `models/llm.py` to experiment with new attention mechanisms, layer types, or normalization techniques.

### Benchmarking
Standardize your experiments by setting specific token targets or model sizes. Use the provided tools in `utils/` to ensure reproducibility.

---

## 💻 GPU Resources

If you need access to GPUs, consider the following options:

- **Lightning AI**: Fast and easy setup with L4 GPUs.
- **Google Colab**: Good for small experiments (T4/A100).
- **VastAI / Salad**: Affordable GPU rentals for larger runs.

**Tip**: If the model doesn't fit in your GPU memory, you can **reduce the model size** (e.g., `batch_size`, `n_layer`, or `n_embd` in `configs/llm_config.py`).
