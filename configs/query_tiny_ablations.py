"""Query / W_Q tweaks at the Tiny1M3M tier (~0.94M params · 3M tokens).

Tiny-tier mirror of the Screen10M20M query zoo in configs/llm_config.py.
Same lever flags, cheapest scale — fast idea filter, 1 seed.
Control is Tiny1M3MConfig (added automatically by scripts/run_research.py).
Folder: query_tiny.
"""
from dataclasses import dataclass

from configs.llm_config import Tiny1M3MConfig


# ---- Batch 1: scoring (Q1-Q4) ----
@dataclass
class Tiny1M3MAlibiBiasConfig(Tiny1M3MConfig):
    """Q1 — ALiBi-style per-head distance bias. scores += -m_h·(i-j)."""
    use_alibi_bias: bool = True


@dataclass
class Tiny1M3MQTempTokenConfig(Tiny1M3MConfig):
    """Q2 — Token-conditioned per-head Q temperature. Q *= (1 + tanh(x·w_h))."""
    use_q_temp_token: bool = True


@dataclass
class Tiny1M3MCosineAttnConfig(Tiny1M3MConfig):
    """Q3 — Cosine attention. L2-normalize Q,K; per-head learnable τ."""
    use_cosine_attn: bool = True


@dataclass
class Tiny1M3MQKBilinearConfig(Tiny1M3MConfig):
    """Q4 — Per-channel relevance. score = Q^T diag(d_h) K (d_h init 1)."""
    use_qk_bilinear: bool = True


# ---- Batch 2: flagship + positional (Q5-Q7) ----
@dataclass
class Tiny1M3MTalkingHeadsQConfig(Tiny1M3MConfig):
    """Q5 — Talking-heads on Q. learned n_h×n_h M on attention logits pre-softmax."""
    use_talking_heads_q: bool = True


@dataclass
class Tiny1M3MPerHeadRopeBaseConfig(Tiny1M3MConfig):
    """Q6 — Per-head learnable RoPE base. θ_h init = global base."""
    use_per_head_rope_base: bool = True


@dataclass
class Tiny1M3MPartialRotaryConfig(Tiny1M3MConfig):
    """Q7 — Partial rotary. Rotate only 50% of Q/K dims."""
    partial_rotary_p: float = 0.5


# ---- Batch 3: exotic (Q8-Q10) ----
@dataclass
class Tiny1M3MQExpansionConfig(Tiny1M3MConfig):
    """Q8 — Multi-query expansion. Q += W·x (zero-init W; step-0 baseline)."""
    use_q_expansion: bool = True


@dataclass
class Tiny1M3MDecoupledContentPosConfig(Tiny1M3MConfig):
    """Q9 — Decoupled content/position attention (DeBERTa-style)."""
    use_decoupled_content_pos: bool = True


@dataclass
class Tiny1M3MAntisymQKConfig(Tiny1M3MConfig):
    """Q10 — Antisymmetric Q·K coupling. +Q^T S K with skew S (init 0)."""
    use_antisym_qk: bool = True


# ---- Batch 4: query-norm zoo (Q11-Q16) ----
@dataclass
class Tiny1M3MNormPNormConfig(Tiny1M3MConfig):
    """Q11 — Q-side pnorm p=1.5 (Lp norm, outlier-robust)."""
    q_norm_type: str = "pnorm1.5"


@dataclass
class Tiny1M3MNormClipConfig(Tiny1M3MConfig):
    """Q12 — Q-side Winsorized RMSNorm (clip k=3)."""
    q_norm_type: str = "clipnorm3"


@dataclass
class Tiny1M3MNormChannelScaleConfig(Tiny1M3MConfig):
    """Q13 — Q-side ChannelScale (learnable pre-scale)."""
    q_norm_type: str = "channelscale"


@dataclass
class Tiny1M3MNormManhattanConfig(Tiny1M3MConfig):
    """Q14 — Q-side Manhattan (L1 MAD) norm."""
    q_norm_type: str = "manhattan"


@dataclass
class Tiny1M3MNormCenterConfig(Tiny1M3MConfig):
    """Q15 — Q-side Center norm (mean-only, no variance)."""
    q_norm_type: str = "center"


@dataclass
class Tiny1M3MNormNoneConfig(Tiny1M3MConfig):
    """Q16 — Q-side norm disabled. K still normed."""
    q_norm_type: str = "none"


# ---- Batch 5: learnable-param zoo (Q17-Q23) ----
@dataclass
class Tiny1M3MQPerHeadBiasConfig(Tiny1M3MConfig):
    """Q17 — Per-head bias. Q += b_h (per-head×channel) post-RoPE."""
    use_q_per_head_bias: bool = True


@dataclass
class Tiny1M3MQPerChannelGainConfig(Tiny1M3MConfig):
    """Q18 — Per-channel gain. Q *= g_d post-RoPE."""
    use_q_per_channel_gain: bool = True


@dataclass
class Tiny1M3MQHDGainConfig(Tiny1M3MConfig):
    """Q19 — Head×channel gain. Q *= g_hd post-RoPE."""
    use_q_hd_gain: bool = True


@dataclass
class Tiny1M3MQNormGateConfig(Tiny1M3MConfig):
    """Q20 — Norm-gate. per-head scalar σ(a_h·‖x‖+b_h) on Q."""
    use_q_norm_gate: bool = True


@dataclass
class Tiny1M3MQLowRankRefineConfig(Tiny1M3MConfig):
    """Q21 — Low-rank refine. Q += (W1·x)@W2 (zero-init)."""
    use_q_lowrank_refine: bool = True


@dataclass
class Tiny1M3MQLayerScaleConfig(Tiny1M3MConfig):
    """Q22 — LayerScale on Q. Q *= (1 + ls_d) per-channel post-RoPE."""
    use_q_layerscale: bool = True


@dataclass
class Tiny1M3MQSoftplusGainConfig(Tiny1M3MConfig):
    """Q23 — Softplus gain. Q *= softplus(g_h) per-head — always ≥ 0."""
    use_q_softplus_gain: bool = True


# ---- Batch 6: architecture/mixing (Q24-Q29) ----
@dataclass
class Tiny1M3MQHeadMixConfig(Tiny1M3MConfig):
    """Q24 — Head-mix. Q ← Q + Q @ M (M−I init 0) pre-attention."""
    use_q_head_mix: bool = True


@dataclass
class Tiny1M3MQTimeConvConfig(Tiny1M3MConfig):
    """Q25 — Time-conv. 1D conv k=3 over position axis, zero-init."""
    use_q_time_conv: bool = True


@dataclass
class Tiny1M3MQEMASmoothConfig(Tiny1M3MConfig):
    """Q26 — EMA-smooth over position. Q ← α·Q + (1−α)·Q_{t-1}."""
    use_q_ema_smooth: bool = True


@dataclass
class Tiny1M3MQFeatureMapConfig(Tiny1M3MConfig):
    """Q27 — Feature-map attention. NOT identity-init — needs own control."""
    use_q_feature_map: bool = True


@dataclass
class Tiny1M3MQPerTokenRopeConfig(Tiny1M3MConfig):
    """Q28 — Per-token RoPE. Each token's θ via small MLP on x."""
    use_q_per_token_rope: bool = True


@dataclass
class Tiny1M3MQNoiseRegConfig(Tiny1M3MConfig):
    """Q29 — Noise reg. Q += N(0, σ²) training only (learnable σ)."""
    use_q_noise_reg: bool = True
