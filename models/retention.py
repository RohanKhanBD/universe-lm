"""RetNet retention kernel (parallel / training path).

Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei (Microsoft) — "Retentive Network:
A Successor to Transformer for Large Language Models" (arXiv:2307.08621).

Replaces softmax attention with a per-head exponential decay kernel:
    O[t, d] = Σ_{s≤t} γ_h^(t-s) · <Q[t], K[s]> · V[s, d]

The decay γ_h ∈ (0, 1) is per-head and learnable (sigmoid-mapped from a
raw parameter, init γ=0.99). Causal: positions s > t contribute 0.

This is v1 — parallel (training) path only. The chunkwise (linear-
complexity) and recurrent (inference) paths are reserved for v2; the
math is the same, the wiring differs. See
`autoresearch/ideas/004-retnet-retention/plan.md` for the scope decision.

Bit-identical default: this module is never instantiated by the model
in v1 (the `use_retention` flag is reserved). v2 will wire it into
`MultiHeadAttention.forward`.
"""
import math
import torch
import torch.nn as nn


def _inv_sigmoid(p: float) -> float:
    """Inverse sigmoid — the raw value such that sigmoid(raw) = p."""
    return math.log(p / (1.0 - p))


class RetentionKernel(nn.Module):
    """RetNet retention kernel — parallel (training) path.

    Per-head learnable decay γ_h, sigmoid-mapped from a raw parameter
    (init γ=0.99 so the kernel's effective receptive field covers the
    first ~100 positions at init; the param can be tuned per-tier).

    Parameters
    ----------
    d_k : int
        Per-head dimension (D in the math above).
    n_heads : int
        Number of independent decay scalars.
    init_gamma : float
        Initial decay; sigmoid-mapped. Must be in (0, 1).

    Forward
    -------
    Q, K, V : [B, H, T, D] tensors
    Returns O : [B, H, T, D]

    Notes
    -----
    - Causal: O[t] is a sum over s ≤ t only (positions s > t have
      γ^(t-s) with negative exponent → masked to 0).
    - No softmax: the kernel is a linear recurrence, not a normalized
      attention. The score scale is set by `<Q[t], K[s]>` directly,
      with no /sqrt(d_k) (the decay provides the implicit scale).
    - Mask built in log-space: `M[t, s] = exp((t-s) · log(γ))` for
      t≥s, 0 otherwise. Avoids overflowing γ^(t-s) at long T.
    """

    def __init__(self, d_k: int, n_heads: int, init_gamma: float = 0.99):
        super().__init__()
        if not 0.0 < init_gamma < 1.0:
            raise ValueError(f"init_gamma must be in (0, 1); got {init_gamma}")
        self.d_k = d_k
        self.n_heads = n_heads
        # Raw parameter; γ = sigmoid(raw). Init raw so sigmoid(raw) = init_gamma.
        self.gamma_raw = nn.Parameter(
            torch.full((n_heads,), _inv_sigmoid(init_gamma))
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q, K, V: [B, H, T, D]
        if Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError(
                f"Q/K/V shape mismatch: {Q.shape} vs {K.shape} vs {V.shape}"
            )
        if Q.size(-1) != self.d_k:
            raise ValueError(
                f"Last dim {Q.size(-1)} != d_k={self.d_k}"
            )
        B, H, T, D = Q.shape

        # Per-head decay γ_h ∈ (0, 1). Shape [1, H, 1, 1] for broadcast.
        gamma = torch.sigmoid(self.gamma_raw).view(1, H, 1, 1)
        # Log-space mask: M[t, s] = γ^(t-s) for t≥s, 0 otherwise.
        # diff[t, s] = t - s; positive on the lower triangle, negative on
        # the upper. We zero the upper triangle so position t cannot
        # attend to positions s > t (causality).
        idx = torch.arange(T, device=Q.device)
        diff = (idx[:, None] - idx[None, :]).to(Q.dtype)  # [T, T]: diff[t,s] = t-s
        log_gamma = torch.log(gamma.clamp_min(1e-8))  # [1, H, 1, 1]
        mask = torch.where(
            diff >= 0,
            torch.exp(diff[None, None, :, :] * log_gamma),
            torch.zeros(1, 1, T, T, dtype=Q.dtype, device=Q.device),
        )  # [1, H, T, T] — γ is per-head, broadcast over the H axis.

        # O = (Q K^T ⊙ M) V. No /sqrt(d_k): the decay provides implicit
        # scale. No softmax: the kernel is a linear recurrence.
        scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, H, T, T]
        scores = scores * mask
        return torch.matmul(scores, V)  # [B, H, T, D]
