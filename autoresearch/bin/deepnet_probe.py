#!/usr/bin/env python3
"""deepnet_probe.py — E5 understanding (no training): measure how DeepNet-a bounds
the residual-stream magnitude across depth.

DeepNet-a scales every block's sublayer output by a = (2*n_layers)^(-1/2) before
the residual add. Wang et al. 2022 (Thm 1) claim this bounds the residual stream's
growth to O(1) across depth instead of the ~sqrt(L) growth of an un-scaled pre-norm
stack. This probe verifies that claim DIRECTLY on the real model at init: build the
network with and without `use_deepnet_alpha`, forward a random batch, and read off
the per-layer residual-stream RMS. No data, no training, no GPU — CPU-only.

Width is held FIXED (d_model=128) and DEPTH is varied, so the effect we see is the
depth-dependent residual bounding, not a width/N artifact (previews E2). See
autoresearch/DEEPNET-RESEARCH.md.

  python3 autoresearch/bin/deepnet_probe.py
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from configs.llm_config import Ladder8M155MConfig
from models.llm import MinimalLLM

DEPTHS = [4, 8, 16, 30]   # 30 = the 135M release target's depth
BATCH, SEQ = 2, 64
SEED = 0


def make_config(n_layers, deepnet):
    """Fixed-width (d=128) config at a given depth, deepnet on/off."""
    fields = [("n_layers", int, dataclasses.field(default=n_layers)),
              ("use_deepnet_alpha", bool, dataclasses.field(default=deepnet))]
    C = dataclasses.make_dataclass("C", fields, bases=(Ladder8M155MConfig,))
    return C()


def per_layer_rms(cfg):
    """Build the model, forward a random batch, return per-block residual RMS."""
    torch.manual_seed(SEED)
    model = MinimalLLM(cfg).eval()
    blocks = model.transformer_blocks
    rms = []

    def hook(_m, _inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        rms.append(float(x.detach().float().pow(2).mean().sqrt()))

    handles = [b.register_forward_hook(hook) for b in blocks]
    vocab = getattr(cfg, "vocab_size", None) or 49152
    ids = torch.randint(0, vocab, (BATCH, SEQ))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    return rms


def main():
    print("=== DeepNet-a residual-stream bounding (step-0, d_model=128, random batch) ===")
    print("Per-block residual RMS; 'grow' = last/first. DeepNet should keep grow ~flat.\n")
    print(f"{'L':>3} {'alpha':>7} | {'baseline first->last (grow)':>34} | {'deepnet first->last (grow)':>34}")
    print("-" * 84)
    for L in DEPTHS:
        a = (2.0 * L) ** -0.5
        base = per_layer_rms(make_config(L, deepnet=False))
        deep = per_layer_rms(make_config(L, deepnet=True))
        bg = base[-1] / base[0] if base[0] else float("nan")
        dg = deep[-1] / deep[0] if deep[0] else float("nan")
        print(f"{L:>3} {a:>7.3f} | {base[0]:>8.3f} -> {base[-1]:>8.3f} ({bg:>5.2f}x) | "
              f"{deep[0]:>8.3f} -> {deep[-1]:>8.3f} ({dg:>5.2f}x)")
    print("\nReading: if baseline 'grow' rises with L while deepnet stays ~flat, the")
    print("mechanism (depth-bounded residual stream) is confirmed empirically — the")
    print("structural reason DeepNet-a is a depth-driven (H1) candidate, not just an")
    print("intercept tweak. sqrt(L) reference grow:  " +
          "  ".join(f"L{L}:{math.sqrt(L):.2f}" for L in DEPTHS))


if __name__ == "__main__":
    main()
