"""DeBERTa-style disentangled relative-position bias."""

import torch
import torch.nn as nn


class DeBERTaRelativePositionBias(nn.Module):
    """Shared clipped relative-position keys for a content-position cross-term.

    The module stores one zero-init learnable table of relative-position keys,
    shared across layers. The caller supplies the content queries `Q`
    (already normed / RoPE'd), and this module returns the additive logit bias
    `Q_c(H_i) · K_p(P_{i-j})` as a [B, H, T, T] tensor.
    """

    def __init__(self, n_heads: int, d_k: int, max_seq_len: int, clip: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.clip = int(clip)
        self.rel_key = nn.Parameter(torch.zeros(2 * self.clip + 1, n_heads, d_k))
        idx = torch.arange(max_seq_len)
        rel = idx[:, None] - idx[None, :]
        rel = rel.clamp(-self.clip, self.clip) + self.clip
        self.register_buffer("rel_idx", rel, persistent=False)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B, H, T, D]. Returns [B, H, T, T].
        _, _, t, _ = q.shape
        rel = self.rel_idx[:t, :t]
        pos_k = self.rel_key[rel].to(dtype=q.dtype)  # [T, T, H, D]
        pos_k = pos_k.permute(2, 0, 1, 3)  # [H, T, T, D]
        return torch.einsum("bhtd,htsd->bhts", q, pos_k)
