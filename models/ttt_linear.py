"""149 — TTT-Linear (Sun, Yang, et al. 2024, arXiv:2407.04620,
"Learning to (Learn at Test Time): RNNs with Expressive Hidden States",
section 3.2). A drop-in replacement for `nn.Linear` whose weight is
updated per input during the forward pass via a single closed-form
gradient step on a self-supervised reconstruction loss.

Mechanism (paper section 3.2, Batched-Vectorized-Newton style):

    y_slow = F.linear(x, weight, bias)              # standard linear
    diff   = x_target - y_slow                      # (..., out)
    outer  = diff.unsqueeze(-1) * x.unsqueeze(-2)   # (..., out, in)
    norm   = (x * x).sum(-1, keepdim=True).unsqueeze(-2) + eps
                                                    # (..., 1, 1)
    W_f    = weight + lr * outer / norm             # (..., out, in)
    y      = (W_f @ x.unsqueeze(-1)).squeeze(-1)    # (..., out)

`x_target` is a *learned* projection of `x` -- `x_target = target_weight
@ x` -- zero-init (paper section 3.2: "x_target is a learned projection
of x"). The fast weight is forced to compress-and-reconstruct this
target. Zero-init means at step 0 the target is the zero vector and
the closed-form update is exactly zero when `lr = 0` (the default).
At `lr = 0` the forward reduces to `F.linear(x, weight, bias)` --
bit-identical to a plain `nn.Linear` with the same `kaiming_uniform_`
weight.

Why a single-line shortcut (`if lr.item() == 0: return y_slow`): the
fully-vectorized fast path computes `(W_slow + lr * outer / norm) @
x` which mathematically equals `W_slow @ x` when `lr=0`, but two
different matmul kernels may produce different fp32 rounding for the
same input. The `if lr == 0` branch makes step-0 byte-identical to a
plain `nn.Linear` while costing one Python `float` comparison per
forward (cheap; only fires when the learnable `lr` is exactly `0.0`).
After the first optimizer step the lr parameter may become nonzero
and the fast path engages.

The fast path costs O(B * T * out * in) extra FLOPs per layer -- at
tiny1m3m (d_model=64, d_ff=128, T=128) this is a 2-3x FFN compute
cost. Acceptable for a 0.94M model with 92 update steps; not a viable
replacement at scale without a CUDA kernel.

Identity at step 0: `lr=0` (default `ttt_lr_init=0.0`) => forward
returns `y_slow = F.linear(x, weight, bias)` with the same
`kaiming_uniform_` weight as `nn.Linear` => bit-identical to a plain
linear with the same `weight` parameter. The `use_ttt_ffn` flag is
OFF by default so the baseline forward graph is bit-identical (the
`TTTLinear` module is never constructed). See
`autoresearch/ideas/149-ttt-linear/idea.md`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TTTLinear(nn.Linear):
    """Drop-in replacement for `nn.Linear` with a per-input fast-weight update.

    Inherits from `nn.Linear` so the global `_init_weights` hook in
    `MinimalLLM` re-initializes `.weight` (and `.bias` when present) via
    `torch.nn.init.normal_(mean=0, std=0.02)` -- exactly the same RNG
    call the baseline FFN's `nn.Linear` would receive. Without the
    inheritance, `_init_weights` would NOT touch the TTT weight and the
    full-model forward at step 0 would NOT be byte-identical to the
    baseline (different init distributions). With the inheritance,
    `use_ttt_ffn=True` produces the same weights as the baseline at
    init time, AND the `lr.item() == 0` short-circuit in `forward`
    collapses to the standard linear path.

    Args:
        in_features: input dimension (matches `nn.Linear`).
        out_features: output dimension (matches `nn.Linear`).
        bias: whether to include a bias (default False, mirrors the
              FFN up-proj convention in this repo).
        ttt_lr_init: initial value of the per-layer TTT learning rate
              (default 0.0 => bit-identical to `nn.Linear` at step 0).
        eps: numerical stabilizer for `||x||^2`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        ttt_lr_init: float = 0.0,
        eps: float = 1e-5,
    ):
        # Initialize via nn.Linear's normal path so `.weight` is a
        # Parameter with the right shape and `.bias` is registered
        # correctly. nn.Linear.__init__ calls reset_parameters()
        # which uses kaiming_uniform_(std=...) -- this is OVERWRITTEN
        # by `_init_weights` to normal_(0, 0.02) at the model level.
        super().__init__(in_features, out_features, bias=bias)
        self.eps = float(eps)
        # TTT learning rate -- zero-init by default => bit-identical at step 0.
        self.ttt_lr = nn.Parameter(torch.tensor(float(ttt_lr_init)))
        # Learned target projection: x_target = target_weight @ x
        # (paper section 3.2: "x_target is a learned projection of x").
        # Zero-init => at step 0 the target is the zero vector regardless
        # of `lr` (and when `lr=0` the fast-weight update is also zero,
        # so the forward collapses to a plain `nn.Linear`). Trained via
        # backprop once `lr` becomes nonzero.
        self.target_weight = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard linear projection: y = W @ x + b. With `bias=False`
        # this is exactly what the baseline FFN's `nn.Linear` produces.
        y_slow = F.linear(x, self.weight, self.bias)
        # Short-circuit: with `lr=0` (default init) the fast-weight
        # gradient term is exactly zero and the fully-vectorized fast
        # path would still produce `y_slow` mathematically but with a
        # different fp32 rounding kernel. Take the slow path to keep
        # step-0 byte-identical.
        lr = self.ttt_lr
        if lr.item() == 0.0:
            return y_slow
        # Auto-encoding target: learned projection of x. Zero-init at
        # construction; trained via backprop once `lr` becomes nonzero.
        x_target = F.linear(x, self.target_weight)            # (..., out)
        diff = x_target - y_slow                              # (..., out)
        outer = diff.unsqueeze(-1) * x.unsqueeze(-2)          # (..., out, in)
        norm = (x * x).sum(-1, keepdim=True).unsqueeze(-2) + self.eps
                                                              # (..., 1, 1)
        update = lr * outer / norm                            # (..., out, in)
        W_f = self.weight.unsqueeze(0).unsqueeze(0) + update  # (..., out, in)
        y = torch.matmul(W_f, x.unsqueeze(-1)).squeeze(-1)    # (..., out)
        return y


class TTTFeedForward(nn.Module):
    """SquaredReLU FFN whose up-projection is a `TTTLinear`.

    The `up_proj` is replaced with `TTTLinear(d_model, d_ff, bias=False,
    ttt_lr_init=ttt_lr_init)` -- a per-input fast-weight linear. The
    `down_proj` stays a standard `nn.Linear(d_ff, d_model, bias=False)`
    so the FFN-output projection is intact (the TTT update is on the
    up-proj side, which is the standard TTT-Linear placement from the
    paper). The squared-ReLU activation and dropout are unchanged.

    Identity at step 0: `ttt_lr=0` (default) => `TTTLinear.up_proj`
    collapses to `F.linear(x, weight, None)` with the same
    `kaiming_uniform_` weight as the baseline `nn.Linear` =>
    `TTTFeedForward` collapses to a vanilla `SquaredReLUFeedForward`.
    Default off => baseline FFN path bit-identical (the
    `TTTFeedForward` is never built). See
    `autoresearch/ideas/149-ttt-linear/idea.md`.

    Param cost when ON: `TTTLinear` adds one scalar `ttt_lr` and one
    `out x in` zero-init `target_weight` parameter on top of the
    baseline `nn.Linear` weight. At tiny1m3m (d_model=64, d_ff=128)
    that's `1 + 64 * 128 = 8193` extra params per FFN. Default off =>
    no overhead.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        ttt_lr_init: float = 0.0,
    ):
        super().__init__()
        self.up_proj = TTTLinear(
            d_model, d_ff, bias=False, ttt_lr_init=ttt_lr_init
        )
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activation is (max(0, x))^2 -- Primer-style, mirrors
        # SquaredReLUFeedForward.forward. The TTT update lives inside
        # `up_proj`.
        return self.down_proj(
            self.dropout(torch.square(F.relu(self.up_proj(x))))
        )
