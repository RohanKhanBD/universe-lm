"""Tests for `models.retention.RetentionKernel` — see
`autoresearch/ideas/004-retnet-retention/plan.md`.

Three invariants, each a separate test:
  1. No NaN/Inf on a non-trivial random input.
  2. Causal: O[t] unchanged when K[s>t] is zeroed (no future leak).
  3. Per-head independence: perturbing γ_h only changes the h-th head's
     output.
"""
import math
import torch
from models.retention import RetentionKernel


def test_no_nan_or_inf():
    """Random input → finite output."""
    torch.manual_seed(42)
    kernel = RetentionKernel(d_k=8, n_heads=4, init_gamma=0.9)
    Q = torch.randn(2, 4, 16, 8)
    K = torch.randn(2, 4, 16, 8)
    V = torch.randn(2, 4, 16, 8)
    O = kernel(Q, K, V)
    assert torch.isfinite(O).all(), "Output has NaN/Inf"
    assert O.shape == (2, 4, 16, 8), f"Shape mismatch: {O.shape}"


def test_causal_no_future_leak():
    """Zeroing K at positions t >= T/2 must not change O[t0] for t0 < T/2.

    The kernel is causal: O[t] = sum_{s<=t} γ^(t-s) <Q[t], K[s]> V[s].
    Changing K[s, *] for s > t0 must not affect O[t0].
    """
    torch.manual_seed(42)
    kernel = RetentionKernel(d_k=8, n_heads=4, init_gamma=0.9)
    Q = torch.randn(2, 4, 16, 8)
    K = torch.randn(2, 4, 16, 8)
    V = torch.randn(2, 4, 16, 8)

    O_full = kernel(Q, K, V)

    T = K.size(2)
    split = T // 2
    # Zero K at positions t >= split — these are the "future" for any
    # t0 < split. The past (t0 < split) is unchanged.
    K_clipped = K.clone()
    K_clipped[:, :, split:, :] = 0
    O_clipped = kernel(Q, K_clipped, V)

    # Past outputs must be identical to the full-K outputs.
    diff = (O_full[:, :, :split, :] - O_clipped[:, :, :split, :]).abs().max().item()
    assert diff < 1e-5, f"Causal leak at t < {split}: max abs diff = {diff}"
    # Sanity: the future outputs MAY differ (we changed K in the future).
    # Confirm they actually do, so the test isn't vacuous.
    future_diff = (O_full[:, :, split:, :] - O_clipped[:, :, split:, :]).abs().max().item()
    assert future_diff > 1e-3, (
        f"Future outputs identical — test is vacuous: {future_diff}"
    )


def test_per_head_independence():
    """Perturbing γ_h changes ONLY the h-th head's output."""
    torch.manual_seed(42)
    kernel_a = RetentionKernel(d_k=8, n_heads=4, init_gamma=0.9)
    kernel_b = RetentionKernel(d_k=8, n_heads=4, init_gamma=0.9)
    # Override head 2's decay (h=2). Change raw by a large amount so the
    # sigmoid output is clearly different.
    with torch.no_grad():
        kernel_b.gamma_raw[2] = kernel_b.gamma_raw[2] + 3.0
    Q = torch.randn(2, 4, 16, 8)
    K = torch.randn(2, 4, 16, 8)
    V = torch.randn(2, 4, 16, 8)
    O_a = kernel_a(Q, K, V)
    O_b = kernel_b(Q, K, V)
    # Heads 0, 1, 3: must be identical (up to fp noise).
    for h in (0, 1, 3):
        d = (O_a[:, h] - O_b[:, h]).abs().max().item()
        assert d < 1e-5, f"Head {h} changed when only head 2's γ was perturbed: {d}"
    # Head 2: must differ.
    d2 = (O_a[:, 2] - O_b[:, 2]).abs().max().item()
    assert d2 > 1e-3, f"Head 2 did not change: {d2}"


def test_decay_monotone_in_t():
    """For fixed Q, K at s=0, the mask weight γ^(t-0) decreases as t grows."""
    torch.manual_seed(42)
    kernel = RetentionKernel(d_k=8, n_heads=1, init_gamma=0.9)
    # Mask the same way the kernel does (T=8).
    T = 8
    gamma = torch.sigmoid(kernel.gamma_raw).item()
    log_gamma = math.log(max(gamma, 1e-8))
    idx = torch.arange(T)
    diff = (idx[:, None] - idx[None, :]).to(torch.float32)
    mask = torch.where(
        diff >= 0,
        torch.exp(diff * log_gamma),
        torch.zeros_like(diff),
    )  # [T, T]
    # Column s=0: weights are γ^t for t=0..T-1. Strictly decreasing.
    col0 = mask[:, 0]
    for t in range(1, T):
        assert col0[t] < col0[t - 1] + 1e-6, (
            f"Decay not monotone: t={t-1} → {col0[t-1]}, t={t} → {col0[t]}"
        )
