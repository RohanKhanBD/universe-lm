from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LLMConfig:
    """Default legacy tuned large preset: 88,630,528 parameters."""

    # Model architecture (88M Params)
    d_model: int = 512       
    n_heads: int = 8         
    n_layers: int = 22
    d_ff: int = 2048         
    
    # GQA parameters
    n_kv_heads: int = 4      
    
    # Data params
    # Use the pre-built dataset downloaded as described in the README
    # (`python data/download_hf_data.py`). The repo author recommends NOT
    # changing the data or max_seq_len. If you think you need to, ASK THE
    # REPO AUTHOR FIRST — it is not recommended. The downloaded data is
    # chunked at seq_len 2048, which the RoPE cache depends on; a mismatch
    # causes runtime errors.
    max_seq_len: int = 2048  # do not change; matches the downloaded data
    vocab_size: int = 49152

    # Low-rank embedding factorization (ALBERT-style). None = full vocab x d_model
    # embedding (default, current behavior). An int r factorizes it into
    # (vocab x r) @ (r x d_model), freeing params to reallocate into transformer
    # depth/width at a fixed total budget. lm_head stays tied to the factorization.
    emb_rank: Optional[int] = None
    # Optional additive low-rank output adapter. This keeps the cheap factorized
    # input embedding but gives the tied softmax a small independent correction.
    # None = baseline tied head only.
    output_adapter_rank: Optional[int] = None

    # Residual-stream levers (both default off; identity/baseline-initialized, so
    # "off" reproduces the current model bit-for-bit).
    # #20 embed residual: re-inject the token embedding x0 each block via a learnable
    #   per-dim mix x = m0*x + m1*x0 (m0 init 1, m1 init 0). Fights representation
    #   drift as depth grows. Costs 2*d_model params/block.
    use_embed_residual: bool = False
    # #22 zero-init resid: zero-init the attention O-projection + FFN down-projection
    #   so each block is an exact identity at step 0 (clean signal propagation through
    #   the deep stack). Zero extra params — purely an init change.
    zero_init_resid: bool = False
    # #27 SmearGate: add a learned per-channel amount of the previous token's
    # embedding before the transformer stack. Causal, zero-init, costs d_model.
    use_smear_gate: bool = False
    # 159 — Embedding pre-LayerNorm (LLaMA 3 / Gemma 2 / Mistral /
    # Qwen 2.5 pattern). Apply a single `nn.LayerNorm(d_model)` on the
    # scaled token embedding right before the first transformer block.
    # Default off → baseline path bit-identical (the LN module is
    # never built, the forward branch is never taken). When on, the
    # LN params are init to `weight = std(x_post)`, `bias = mean(x_post)`
    # (empirical, computed at construction) so `LN(x_post) ≈ x_post`
    # at step 0 within fp32 rounding noise — the model starts as
    # exactly the baseline residual stream and the LN earns its
    # normalisation during training. Cost: 2·d_model params (128 at
    # tiny1m3m — negligible). See
    # `autoresearch/ideas/159-emb-layernorm/idea.md`.
    use_emb_layernorm: bool = False
    # #23 U-Net skips: add zero-init learned bridges from early block outputs to
    # mirrored late blocks. Helps deep narrow stacks preserve early lexical detail.
    use_unet_skips: bool = False
    # Number of early-to-late bridge pairs. None = n_layers // 2 (full U).
    # Must be <= n_layers // 2 (bridges read from un-saved early activations
    # otherwise). Only active when use_unet_skips=True.
    unet_skip_count: Optional[int] = None
    # Gate parameterization for U-Net skips. "raw" applies the gate directly
    # (`x = x + gate * skip`); "sigmoid" wraps as `x = x + sigmoid(gate) * skip`
    # (modded-nanoGPT speedrun style). Default "raw" + init 0.0 reproduces the
    # current behaviour bit-for-bit.
    unet_gate_type: str = "raw"
    # Initial gate parameter value (broadcast to (skip_count, d_model)).
    # With unet_gate_type="raw", 0.0 means no contribution at step 0 (default,
    # current behaviour). With unet_gate_type="sigmoid", -1.5 matches the
    # speedrun: sigmoid(-1.5) ~= 0.18 of the early activation flows in at step 0.
    unet_gate_init: float = 0.0
    # RMSNorm each early-layer skip before the gated add. Aims to keep the bridge
    # contribution scale-stable as the residual stream grows over training. Off
    # reproduces the current behaviour. Only active when use_unet_skips=True.
    unet_bridge_norm: bool = False
    # #28 Attention output gate: zero-init per-head multiplier on attention output.
    # Starts as exact baseline via output *= (1 + gate).
    use_attn_output_gate: bool = False
    # Value-channel gate: zero-init per-head, per-channel multiplier on V.
    # Starts as exact baseline via V *= (1 + gate). Distinct from the
    # scalar per-head output gate above because it acts before the
    # weighted sum.
    use_value_channel_gate: bool = False
    # Attention-output channel gate: zero-init per-head, per-channel
    # multiplier on the post-AV head outputs. Starts as exact baseline via
    # output *= (1 + gate). This is the channelwise sibling of
    # use_attn_output_gate.
    use_attn_output_channel_gate: bool = False
    # 152 — Per-head attention logit bias (PaLM 2 §arch, OLMo 2).
    # Learnable `b_h ∈ R^H` added to attention logits pre-softmax.
    # Init 0 → step-0 byte-identical to baseline. NB: per-head
    # *scalar* bias cancels in softmax over the key axis for all
    # subsequent steps too (per-(b,h,t) `e^{b_h}` factor cancels in
    # the per-row normalizer); the experiment is a recorded null.
    # Default off → baseline path bit-identical. See
    # `autoresearch/ideas/152-attn-logit-bias/idea.md`.
    use_attn_logit_bias: bool = False
    # 155 — Per-head learnable attention temperature
    # (PaLM 2 §arch, OLMo 2, Gemma 2). Replace the standard
    # `1/sqrt(d_k)` attention scale with a per-head learnable
    # scalar `τ_h ∈ R^H` so the per-head logit scale becomes
    # `Q_h K_h^T * τ_h`. Init `τ_h = 1/sqrt(d_k)` exactly ⇒
    # `Q_h K_h^T * (1/sqrt(d_k))` ≡ `Q_h K_h^T / sqrt(d_k)`
    # (bit-identical to the standard pre-softmax scale) at step 0.
    # Each head can then adjust its own temperature during
    # training — heads wanting sharper focus can lower `τ_h`,
    # heads wanting broader context can raise it. Cost: H
    # scalars/layer (4 at tiny1m3m, total 48 — negligible). Default
    # off → baseline path bit-identical (no Parameter registered,
    # no branch taken). See
    # `autoresearch/ideas/155-per-head-temp/idea.md`.
    use_per_head_temp: bool = False
    # 161 — Per-layer learnable attention temperature. Replace the
    # standard `1/sqrt(d_k)` attention scale with a per-layer
    # learnable scalar `τ_l ∈ R^{n_layers}` so the per-layer logit
    # scale becomes `Q_h K_h^T * τ_l` (the same scale factor across
    # all heads in a layer, but different across layers). Init
    # `τ_l = 1/sqrt(d_k)` exactly ⇒ `Q_h K_h^T * (1/sqrt(d_k))`
    # ≡ `Q_h K_h^T / sqrt(d_k)` (bit-identical to the standard
    # pre-softmax scale) at step 0. Each layer can then adjust
    # its own temperature — early layers can broaden attention,
    # late layers can sharpen it. Cost: 1 scalar/layer
    # (12 at tiny1m3m, total 12 — negligible). Forces the manual
    # attention path so SDPA's flash/efficient backends don't
    # perturb step-0 numerics. Distinct from `use_per_head_temp`
    # (155): per-head varies WITHIN a layer (H scalars/layer),
    # per-layer varies ACROSS layers (1 scalar/layer). The two
    # are orthogonal axes. Default off → baseline path bit-
    # identical (no Parameter registered, no branch taken). See
    # `autoresearch/ideas/161-dyt-temp/idea.md`.
    use_per_layer_temp: bool = False
    # 160 — Per-head RMS gain on the attention output (Gemma 2 /
    # Qwen 2.5). After the AV product and softmax aggregation, multiply
    # each head's output `o_h = (A·V)_h ∈ R^{T×d_k}` by a learnable
    # scalar `g_h ∈ R^H` so each head controls the magnitude of its
    # contribution to the residual stream without changing direction.
    # Init `g_h = 1.0` exactly ⇒ `o_h *= 1 = o_h` byte-identical to
    # baseline at step 0. Distinct from `use_attn_output_gate` (reparam
    # `(1+g_h)` with g_h=0 init): that one starts at 1.0 but its
    # magnitude reparam has the gradient concentrated in `g_h`; this
    # one is a direct `g_h` multiplier so the magnitude *and*
    # gradient are both `g_h`. Distinct from `use_layerscale`/
    # `use_layer_scale`: those operate on the residual add after the
    # O projection; this lever fires on the *head* dimension before
    # the O projection, normalizing per-head output magnitudes
    # independently of the residual stream. Cost: H scalars/layer
    # (4 at tiny1m3m, total 48 — negligible). Default off → baseline
    # path bit-identical (no Parameter registered, no branch taken).
    # See `autoresearch/ideas/160-rms-gain-per-head/idea.md`.
    use_head_gain: bool = False
    # #107 Exclusive self-attn: subtract the component of the attention
    # output that lies along the current token's value vector. Zero-init
    # per-head coefficient → step-0 is baseline; default off keeps the
    # existing attention path bit-identical.
    use_exclusive_self_attn: bool = False
    # #21 LayerScale: zero-init per-channel scales on attention and MLP residual
    # outputs. Starts as exact baseline via branch *= (1 + gate).
    use_layerscale: bool = False
    # 142 — LayerScale (Touvron et al. 2021, arXiv:2103.17239). Per-channel
    # learnable diagonal scale `gamma ∈ R^{d_model}` on each sublayer's
    # residual branch. Direct form `x = x + gamma * sub_block(x)` (NOT the
    # reparam `(1+g)` form used by the existing `use_layerscale` flag above).
    # Init `gamma = layer_scale_init * ones(d_model)` (default 1e-4) → at
    # step 0 the residual contribution is `1e-4 × sub_block(x)`, four
    # orders of magnitude smaller than the residual stream magnitude, so
    # the val loss at step 0 is within fp32 noise of the baseline (the
    # "soft warmup" the paper specifies). Distinct from the existing
    # `use_layerscale` lever: that one is reparam `(1+g)` with g=0 init
    # (identity at step 0); this one is direct `g·sub_block` with g=ε init
    # (soft-warmup). Per-channel vs scalar (ReZero) is the headline
    # architectural novelty — see `autoresearch/ideas/142-layerscale/`.
    use_layer_scale: bool = False
    layer_scale_init: float = 1e-4
    # 130 — ReZero (Bachlechner et al. 2020, arXiv:2003.04887): per-sublayer
    # learnable scalar α on the residual add (one for attention, one for
    # FFN), init 0 ⇒ the entire stack is the identity at step 0 and each
    # layer earns its residual contribution during training. Replaces
    # the baseline add `x = x + f(x)` with `x = x + α·f(x)`. Off by
    # default → baseline path bit-identical. Cost: 2 scalars/block
    # (n_layers × 2 = 24 scalars at tiny1m3m; negligible). See
    # `autoresearch/ideas/130-rezero/idea.md`.
    use_re_zero: bool = False
    # #29 value embeddings (speedrun records 55/63): inject the (factorized) token
    # embedding into attention V at every layer via a tiny per-layer projection,
    # zero-inited so step 0 == baseline. Reuses the existing rank-r table as the
    # source, so cost is only ~r*kv_size/layer (~56k total) — stays in budget.
    use_value_embed: bool = False
    # #30 query embeddings: same trick on Q. Tests whether V's win was
    # V-specific or generalizes to "token identity straight into attention."
    use_query_embed: bool = False
    # #31 key embeddings: same trick on K. K goes through RoPE downstream,
    # so the projection's term is positionally rotated — different operating
    # point from V (no RoPE) or Q (also RoPE'd).
    use_key_embed: bool = False
    # #33 output embeddings: same trick, but applied AFTER the O projection
    # (output side of attention, not input side). The token's raw embedding
    # bypasses the attention computation entirely and lands straight in
    # the residual stream. This is the modded-nanogpt speedrun "value
    # embeddings" position — tests whether V-embed's win is V-specific or
    # is "any token-signal-to-residual" wins.
    use_output_embed: bool = False
    # #37 per-head Q-gain: a learnable per-head scalar that multiplies
    # the Q vector after norm+RoPE. Zero-init = baseline at step 0.
    # Equivalent to a per-head temperature on attention scores.
    # Non-embed lever: changes the attention math, not the inputs.
    use_q_gain: bool = False
    # #42 per-head K-gain: symmetric to Q-gain. Multiplies K after
    # norm+RoPE. Tests if K side also benefits from per-head scaling
    # and if V+q+k_gain beats V+q_gain.
    use_k_gain: bool = False
    # #45 deep value embeddings: 2-layer non-linear V projection.
    # V += GELU(ve @ W1) @ W2. Tests if the linear V-embed (#29) has
    # saturated at a single projection, or whether a non-linear
    # "bottleneck" V-embed has more capacity. Mutually exclusive
    # with use_value_embed. Hidden dim is deep_value_embed_hidden
    # (default 96 = 2× emb_rank for Screen10M20M).
    use_deep_value_embed: bool = False
    deep_value_embed_hidden: Optional[int] = None
    # #47 FFN embeddings: add a learned projection of the factorized
    # token embedding to the FFN input. Different position from
    # V-embed (in attention) and O-embed (post-O residual). Tests
    # whether the V-embed win is about attention content or about
    # residual content. The FFN now has direct access to token
    # identity without going through attention. Cost: 24 × (d_model
    # 144 × emb_rank 48) = 165,888 extra params (~2.1%).
    use_ffn_embed: bool = False
    # 153 — Squared-ReLU FFN activation (So et al. "Primer", arXiv:2109.08668,
    # 2021). When True, swap the FFN's activation for `relu2(x) = x *
    # F.relu(x)` (equivalently `(max(0, x))^2`) regardless of `ffn_variant`.
    # Two-projection shape (up_proj, down_proj, dropout) — same param count
    # as `SquaredReLUFeedForward` so the lever is purely the activation
    # change. At init with normal-distributed pre-activations, both GELU
    # and `ReLU²` produce zero-mean, similar-variance outputs — the first
    # forward differs by < 1e-3 in fp32 max-abs-diff (well inside the
    # harness tolerance for non-bit-identical flags). Default off → no
    # `ReLU2FeedForward` is constructed and the standard `ffn_variant`
    # path runs bit-identical to baseline at step 0. See
    # `autoresearch/ideas/153-relu2-ffn/idea.md`.
    use_relu2_ffn: bool = False
    # 157 — Depthwise Conv inside FFN (ConvBERT/ConvNeXt-style, Jiang
    # et al. 2020 arXiv:2008.02496; Woo et al. 2020). When True, each
    # block builds a `ConvFFN(d_model, kernel=k)` that applies a
    # symmetric depthwise Conv1d to the FFN output (post-FFN, pre-
    # residual-add). Conv weights are identity-initialized (center tap
    # = 1, rest = 0) so the conv is a strict identity at step 0 ⇒
    # baseline path bit-identical when the flag is off (the `ConvFFN`
    # module is never built, the forward branch is never taken).
    # `conv_ffn_kernel` defaults to 3 (spec pin); valid range is odd
    # integers ≥ 3. Differs from 143-shortconv by placement (post-FFN
    # vs pre-attention) and causality (symmetric vs causal). Cost:
    # n_layers × (kernel × d_model) extra params (~2.3K at tiny1m3m
    # with k=3, +0.25%). See `autoresearch/ideas/157-conv-ffn/idea.md`.
    use_conv_ffn: bool = False
    conv_ffn_kernel: int = 3
    # 163 — Post-Attention V-Mix Depthwise Convolution (Poli et al.
    # "Hyena", 2023, arXiv:2302.10866). After the attention output
    # is computed (post-SDPA, post-reshape [B,T,H,D]→[B,T,d_model],
    # pre-W_O projection), apply a symmetric depthwise Conv1d on the
    # time axis over the post-attention tensor. Conv weights are
    # built as raw `nn.Parameter(zeros(d_model, 1, k))` with center
    # tap = 1.0 set inline ⇒ the conv is a strict identity at step 0.
    # Padding = k//2 symmetric (causal+future) — the attention
    # sublayer has already integrated the full causal context, so the
    # conv may look at both neighbors. `v_mix_conv_kernel` defaults to
    # 3 (spec pin); valid range is odd integers ≥ 3. Third axis of
    # the 3-axis locality test (143-shortconv pre-attn, 157-conv-ffn
    # post-FFN, 163-v-mix-conv post-attention on V). Default off →
    # baseline path bit-identical (no Parameter registered, no
    # forward branch taken). Cost: n_layers × k × d_model extra
    # params (12 × 3 × 64 = 2,304 at tiny1m3m, +0.25%). See
    # `autoresearch/ideas/163-v-mix-conv/idea.md`.
    use_v_mix_conv: bool = False
    v_mix_conv_kernel: int = 3
    # #49 QK-norm-post-RoPE: apply RMSNorm to Q,K AFTER RoPE (modded-
    # nanogpt variant) instead of the default BEFORE RoPE. Flag-only,
    # no extra params. The post-RoPE norm constrains post-RoPE Q,K
    # magnitudes per head, which can help with attention score
    # stability. Mathematically different from pre-RoPE norm.
    use_qk_norm_post_rope: bool = False
    # #51 sliding-window attention: replace the full causal mask with
    # a local causal window of width `sliding_window_size`. Flag-only,
    # no extra params. Tests whether the attention *pattern* (not just
    # the inputs) has headroom at this scale — i.e. whether most of the
    # useful long-range signal can be replaced by a short window. If
    # window matches our 2k seq_len's natural coherence, this is a
    # capacity-shaping lever. If not, it caps the model's context.
    use_sliding_window: bool = False
    sliding_window_size: int = 512
    # #53 NoPE: skip RoPE entirely. Flag-only, no extra params. The
    # purest test of whether positional information is load-bearing
    # at this scale. If NoPE ≈ baseline, position is mostly conveyed
    # by the causal mask + token identity (and our token identity
    # injection via V-embed may be partially substituting for RoPE).
    # If NoPE << baseline, RoPE is critical and there's no slack.
    use_nope: bool = False
    # FIRE positional encoding (Li et al., NeurIPS 2023, arXiv:2306.02613):
    # content-aware additive logit bias `bias(t,s) = γ(|t-s|) · f([φ(x_t); φ(x_s)])`
    # with fixed γ (Lp-norm kernel) and small per-head learnable φ/f. Drop-in
    # for RoPE when ON. Default off → baseline path bit-identical. f is
    # zero-init so step-0 bias = 0 even with the flag on.
    # See autoresearch/ideas/009-fire-pe/plan.md.
    use_fire_pe: bool = False
    fire_pe_d_phi: int = 4
    # 013 — CoPE (Golovneva et al. 2024, arXiv:2405.18719, Meta):
    # content-aware positional bias replacing RoPE. Position offset
    # between query i and key j is the count of "important" tokens
    # (those with high content dot-product to a per-head learned probe)
    # in [j, i] — not the literal index distance. Replaces RoPE when
    # ON: the Rotary construction is gated off, the Q/K RMSNorm still
    # runs (magnitude-stabilizer role), and the CoPE bias is added to
    # attention scores in the manual branch. Default off → baseline
    # path bit-identical. Probe init N(0, 0.02) (mirrors FIRE's per-
    # head content projection init at `models/fire_pe.py:60`); τ=0
    # pinned (one-seed-only rule forbids the τ sweep). A/B is
    # "FIRE + CoPE" vs "FIRE" — stacked lever, not a replacement.
    use_cope: bool = False
    # 020 — Forgetting Transformer (FoX, Lin et al. 2025, arXiv:2503.02130):
    # per-head, per-token learnable forget gate that multiplies the
    # attention matrix element-wise after softmax, then row-renormalizes.
    # Conservative extension of softmax attention (the softmax stays; the
    # projection stays; the V path is unchanged). Strictly orthogonal to
    # FIRE (which is additive on logits) — FIRE changes *which* key wins,
    # FoX changes *how much mass* even the winners keep. Identity-init:
    # W_f^h = 0, b_f^h = +10 (pinned at `models/fox.py:FOX_BF_INIT`) →
    # f ≈ 0.99995 at init, D[i,j] within 9% of 1 over the full T=2048
    # context, so the post-softmax decay barely fires at step 0 and the
    # model has to *learn* to forget from scratch. Default off → baseline
    # path bit-identical. Forces the manual attention path (the
    # post-softmax multiply can't go through SDPA's flash kernel). See
    # `autoresearch/ideas/020-forgetting-attn/plan.md`.
    use_fox: bool = False
    # 022 — Softpick (Zuhri/Fuadi/Aji 2025, arXiv:2504.20966):
    # rectified-softmax attention `softpick(x) = relu(exp(x)−1) /
    # (Σ|exp(x)−1| + ε)`. Drop-in for `torch.softmax` in the FIRE
    # manual-path branch. Permits zero total attention mass → kills
    # the attention-sink pathology without adding a learnable sink
    # token. ε=1e-6 pinned (paper default). `exp−1` is computed in
    # fp32 then cast back, otherwise large positive scores overflow
    # in fp16/bf16. No new params, no schedule, no init tuning.
    # Default off → softmax baseline path bit-identical. See
    # `autoresearch/ideas/022-softpick-attention/plan.md`.
    use_softpick: bool = False
    # 025 — Scalable-Softmax (SSMax, Nakanishi 2025, arXiv:2501.19399):
    # per-head learnable scalar s_h that scales the attention logits
    # by `s_h · log(n)` pre-softmax, where n is the per-query causal
    # key count. Restores per-position sharpness at long range so the
    # softmax distribution does not flatten toward uniform as context
    # grows. Drop-in to the manual attention branch; forces the
    # manual path (score-side tweak, can't go through SDPA's flash
    # kernel). Init s_h = 1.0 — the paper's natural starting point
    # and an effective identity on the operator. NOT bit-identical to
    # vanilla softmax at flag-on, n > 1: the log(n) scaling IS the
    # mechanism (paper §3.1), so the step-0 numerical drift is
    # explicitly justified, not a bug. Default off → baseline path
    # bit-identical. See
    # `autoresearch/ideas/025-scalable-softmax/plan.md`.
    use_ssmax: bool = False
    # 023 — Canon conv (Griffin / Mamba local-mixing, De/Smith/Fernando
    # 2024, arXiv:2402.19427; Allen-Zhu et al. Canon-layer line):
    # one causal depthwise Conv1d (kernel=3, left-pad 2) on the
    # residual stream per block, immediately before the attention
    # sublayer's pre-LN. Single scalar output gate `g` per block
    # init 0 → step-0 ≡ no-conv baseline. Pre-LN read (no extra
    # norm on the conv path). Strictly orthogonal to the attention-
    # side levers (FIRE/CoPE/FoX/Softpick all live inside the
    # attention computation; canon conv is on the residual stream
    # itself). Default off → baseline path bit-identical. See
    # `autoresearch/ideas/023-canon-conv/plan.md`.
    use_canon_conv: bool = False
    # 143 — ShortConv (Hyena ShortConv variant, Poli/Massaroli et al.
    # 2023, arXiv:2302.10866): one identity-init depthwise causal
    # Conv1d per block on the residual stream, immediately before the
    # attention sublayer's pre-LN (same placement as CanonConv 023).
    # Weights are identity-initialized (center tap = 1, rest = 0) and
    # a per-block scalar output gate `g` is init 0 → step-0 ≡
    # no-conv baseline (the conv has identity init but the gate
    # scales the contribution to 0, so `x = x + 0·x = x` at step 0).
    # The conv is a *pre-attention* local aggregator: cheap k-neighbor
    # context before the global attention pass. Differs from
    # CanonConv by (a) the identity-init weights (vs Kaiming-uniform)
    # and (b) the parameterizable kernel `short_conv_kernel` (3 or 4).
    # Pre-LN read. Default off → baseline path bit-identical (the
    # `ShortConv1D` module is never built, the forward branch is
    # never taken). See `autoresearch/ideas/143-shortconv/idea.md`.
    use_short_conv: bool = False
    short_conv_kernel: int = 3
    # 021 — Value Residual Learning (Zhou/Wu/Jiang 2024,
    # arXiv:2410.17897). Cross-layer V shortcut: stash the projected V
    # from layer 0 (post-W_V, post-GQA repeat_interleave, post-transpose,
    # shape `[B, n_heads, T, d_k]`); in every later layer l > 0, blend
    # `V_l ← (1 - λ_l)·V_l + λ_l·V_1` BEFORE `attn_weights @ V`, with
    # `λ_l = nn.Parameter(torch.zeros(()))` per-block on MHA. λ=0 init ⇒
    # `V_l = V_l` bit-identical to baseline at step 0; the model has to
    # *learn* to mix in the cross-layer shortcut. `.detach()` on the
    # layer-0 stash so the layer-l blend's gradient does not flow back
    # into layer-0 W_V (each layer's W_V trains on its own attention
    # path). Distinct from the closed V/Q/K/O *embedding* axis (input-
    # side projection scaling) and from every active attention-side lever
    # (020-FoX is post-softmax A·D, 022-softpick is the softmax swap,
    # 024-gated-attn is post-AV o_h gate, 025-SSMax is logit temperature).
    # 021 is the only lever on the *projected V stream*, and the cross-
    # layer formulation is what makes it orthogonal to the closed
    # value-embed axis. Default off → baseline path bit-identical
    # (no `nn.Parameter` created, no stash, no blend). See
    # `autoresearch/ideas/021-value-residual/plan.md`.
    use_value_residual: bool = False
    # #55 layer tying (ALBERT-style): when tie_layer_groups=N, every
    # group of N consecutive blocks shares weights. The model creates
    # n_layers // N unique blocks and the forward pass cycles through
    # them. group_size=1 (default) is the standard non-tied baseline.
    # group_size=2 means 12 unique blocks for 24 layers — half the
    # unique depth params, with each block seeing two distinct
    # positions. Incompatible with U-Net skips.
    tie_layer_groups: int = 1
    # #63 RoPE base: control the wavelength of the rotary. The default
    # base=10000 is GPT-Neo style; Llama uses 500000 which extends the
    # useful positional range. Tests whether the default decay is
    # hurting at our seq_len=2048 (e.g. a 500k base keeps more
    # headroom for distant positions).
    rope_base: int = 10000
    # #71 logit softcap (Gemma-style): cap logits at ±softcap via
    # `logits = softcap * tanh(logits / softcap)`. A 0.0 value
    # disables it (default). Gemma uses 30.0; we test smaller caps
    # (15.0, 20.0) since our model is smaller. The cap is applied
    # right before the LM head loss, so the gradient is backpropped
    # through the tanh. Real arch change — known stabilizer that
    # can change the loss landscape and unlock better minima.
    logit_softcap: float = 0.0
    # OH4 OutputTemp (OutputHead Batch 2 — see docs/research/output_head/plan.md):
    # divides logits by a learnable scalar τ. τ=1 init = no-op at step 0.
    # 1-D parameter, routes to AdamW. Logit op — flows into eval CE legitimately.
    use_output_temp: bool = False
    # OH5 VocabBias (OutputHead Batch 2 — see docs/research/output_head/plan.md):
    # adds a learnable per-vocab bias b_v to the logits (logits += b_v). b=0
    # init = no-op at step 0. 1-D parameter of size vocab_size, routes to AdamW.
    # Logit op — flows into eval CE legitimately. Re-learns token frequency.
    use_vocab_bias: bool = False
    # 144 — Mixture of Softmaxes (Yang, Chen, et al. 2017,
    # arXiv:1711.03953, "Breaking the Softmax Bottleneck"). When on,
    # replace the single output softmax with `n_mos_components` parallel
    # vocab-sized heads mixed by per-token π = softmax(W_π · h). The
    # mixture output is `P(v) = Σ_k π_k · softmax(W_k · h)[v]`, computed
    # in log space via `logsumexp_k (log π_k + log_softmax(W_k · h))`.
    # The structural lever is the *rank* of the output distribution: a
    # single `softmax(W·h)` has rank ≤ d_model, but a K-mixture has
    # effective rank ≤ K·d_model. Identity at step 0: W_π.weight = 0 and
    # W_π.bias = [+1e4, -1e4, ..., -1e4] ⇒ `softmax(W_π·h) = [1, 0, ...,
    # 0]` exactly in fp32 (the `exp(-2e4)` terms underflow to 0), so
    # `logsumexp` reduces to `log_softmax(W_0 · h)` — bit-identical to
    # the standard tied head. The K fresh heads cost `K·vocab·d_model`
    # params (12.6M at tiny1m3m with K=4 — a sizeable param injection,
    # acknowledged as a confound). Default off → baseline path
    # bit-identical (no MoS module built, no forward-graph branches).
    # See `autoresearch/ideas/144-mos/idea.md`.
    use_mos: bool = False
    n_mos_components: int = 4
    # Forward chunk size (along B*T) for the MoS head. Default 128
    # tokens keeps peak memory well under 1 GiB at tiny1m3m (with
    # K=4, V=49152, fp32). The runner report on round 1 showed K=4
    # with chunk=256 still OOM'd on the RTX 3060 12GB because the
    # downstream `F.cross_entropy(logits.view(-1, V), …)` internally
    # materializes a full (N, V) tensor at fp32 (≈ 3.0 GiB at
    # tiny1m3m). Halving the chunk size to 128 keeps the per-chunk
    # peak ~2× smaller; this combined with K=2 in the
    # `Tiny1M3MMoSConfig` subclass keeps MoS training inside the
    # 12GB envelope. Increase for fewer kernel launches at the cost
    # of more peak memory; decrease further if you hit OOM on
    # smaller GPUs. Only consulted when `use_mos=True`.
    mos_chunk_size: int = 128
    # #72 Tied QK (PaLM-style): Q and K share the same projection
    # matrix. The merged QK is shape [q_size + kv_size, d_model],
    # output is split into Q (q_size) and K (kv_size). Real arch
    # change — PaLM's signature attention choice. The Q and K
    # weights are no longer independent; this is a structural
    # constraint, not a hyperparam.
    use_tied_qk: bool = False
    # #73 Multi-head Latent Attention (MLA, DeepSeek-V2): compress
    # K, V into a low-rank latent of dim `mla_latent_dim`, then
    # up-project per head. Different attention design from the
    # standard projection-per-head. Real arch change.
    use_mla: bool = False
    mla_latent_dim: Optional[int] = None
    # #74 Dilated attention: like SWA but the window consists of
    # every `dilation`-th position (instead of contiguous). Tests
    # whether strided/sparse patterns beat contiguous locality.
    # dilation=1 (default) means contiguous (SWA). dilation=2 means
    # every other position. The window still covers ~`window_size`
    # positions by token count, but spread across a longer range.
    attention_dilation: int = 1
    # #75 Post-norm: instead of pre-norm (norm before attn/ffn),
    # place the norm AFTER the residual addition. The original
    # Transformer used post-norm; pre-norm is the modern default.
    # Tests whether post-norm is a real lever at our depth=24.
    use_post_norm: bool = False
    # #76 embedding scale: the standard code multiplies the
    # token embedding by sqrt(d_model). When set to a value
    # other than -1.0, that value is used instead. -1.0 = use
    # the standard sqrt(d_model). Tests whether the standard
    # scaling is a hidden knob.
    embedding_scale: float = -1.0
    # #77 Q/K dim ratio: by default Q and K have the same dim
    # (d_k). When set != 1.0, K is widened to d_k * qk_k_ratio.
    # Tests whether asymmetric Q/K dims change dynamics.
    qk_k_ratio: float = 1.0
    # #79 LayerNorm vs RMSNorm: the base code uses RMSNorm. Set
    # this flag to use LayerNorm instead (with learned affine).
    # Flag-only — drops in via nn.LayerNorm in place of
    # nn.RMSNorm. Tests whether the choice of norm is a real
    # architecture lever on the best baseline.
    use_layernorm: bool = False
    # #80 Linear attention (Performer-style): replace
    # softmax(QK^T / sqrt(d_k)) V with phi(Q) (phi(K)^T V)
    # where phi(x) = elu(x) + 1 (the standard positive
    # random-feature kernel). Flag-only — different attention
    # math, can be O(n) instead of O(n^2) in the windowed
    # case. Tests whether linear-attention math unlocks a new
    # operating point on the best baseline.
    use_linear_attn: bool = False
    # #86 Interleaved global attention (DeepSeek-V4 hybrid-attention
    # analog): the model is otherwise all-SWA (cheap local context
    # everywhere, like V4's compressed/sparse path). When
    # global_attn_every_k > 0, every k-th layer (1-indexed) drops the
    # sliding window and runs full causal attention instead — a cheap
    # periodic "global" layer (V4's HCA-style global context). 0 = off
    # (every layer uses whatever use_sliding_window says). Requires
    # use_sliding_window=True to have any effect; on a full-attention
    # baseline every layer is already global. Flag-only, no extra
    # params — it only changes which layers see the window mask.
    global_attn_every_k: int = 0
    # #87 Differential Attention (Microsoft DIFF Transformer, adapted for
    # small heads): split each head's d_k in half, compute two softmax
    # attention maps, output map1 - lambda*map2 (learnable per-head
    # lambda). Cancels common-mode attention noise. Needs even d_k.
    use_diff_attn: bool = False
    # #88 NSA-style compressed-global attention (DeepSeek Native Sparse
    # Attention, adapted): local sliding window PLUS a global branch over
    # block-mean-pooled K/V summaries (block size nsa_block). Zero-init
    # per-head gate, so step 0 == the local-attention baseline.
    use_nsa_global: bool = False
    nsa_block: int = 64
    # #89 Hybrid heads (DeepSeek-V4 hybrid attention at head granularity):
    # first half of heads attend locally (sliding_window_size), second half
    # attend over full causal context, every layer. Zero extra params.
    use_hybrid_heads: bool = False
    # #90 Residual-stream normalization type. "rmsnorm" (default) / "layernorm"
    # / invented variants: "peak" (L-inf), "manhattan" (L1), "squash" (DyT-style
    # tanh, reduction-free), "center" (mean-only), "manifold" (fractional-power
    # RMS with learnable strength rho). See models/layers.make_norm.
    norm_type: str = "rmsnorm"
    # #91 Robust QK-norm: norm applied to Q,K before the attention dot product
    # (default "rmsnorm" == current behaviour; e.g. "pnorm1.5" for outlier-
    # robust attention logits). #92 Robust V-norm: norm applied to V before the
    # softmax-weighted sum ("" / "none" = off; e.g. "pnorm1.5").
    qk_norm_type: str = "rmsnorm"
    v_norm_type: str = ""
    # #97 Multi-scale heads: each head a different sliding-window size
    # (geometric spread around sliding_window_size). #98 Parallel block
    # (PaLM/GPT-J): attention + FFN read one shared norm and sum into the
    # residual, instead of running sequentially.
    use_multiscale_heads: bool = False
    use_parallel_block: bool = False
    # 162 — Q-Only RMSNorm (asymmetric QK pre-softmax normalization). Apply
    # RMSNorm to Q only, leave K raw. nn.RMSNorm weight=1, bias=0 init
    # ⇒ step-0 ≡ RMSNorm-rescaled Q (spec-allowed fp32 max-abs-diff < 1e-3
    # tolerance, same trade-off as 159-emb-layernorm). Default off ⇒ no
    # module built, baseline path bit-identical. See
    # autoresearch/ideas/162-q-only-norm/idea.md.
    use_q_only_norm: bool = False
    # #99 Attention sink slot (softmax-off-by-one): append a zero K/V the query
    # can attend to, so it isn't forced to dump probability on a real token.
    use_attn_sink: bool = False
    # 017 — Sub-LN / Sandwich block (Wang et al. 2022, DeepNet §3.1; Shleifer
    # et al. 2021, NormFormer): wrap each sublayer output with a fresh
    # `nn.LayerNorm(d_model)` (γ=1, β=0 init → identity at step 0) so the
    # pre-norm baseline path stays bit-identical when the flag is off.
    # On the pre-norm path, `y = x + LN_post(Sublayer(LN_pre(x)))` for both
    # attention and FFN. The pre-LN remains whatever norm_type/use_layernorm
    # selected; the post-LN is always `nn.LayerNorm` (the residual-stream
    # re-bounding role, separate from the magnitude-stabilizing pre-LN).
    use_sub_ln: bool = False
    # 111 — DropPath / Stochastic Depth (Huang et al. 2016, arXiv:1603.09382).
    # Per-block Bernoulli gate during training: with probability `1 - p_l`
    # skip the whole block (residual update `x ← x`), with probability `p_l`
    # keep and rescale the block's contribution by `1/p_l` so the expected
    # residual magnitude is preserved. `p_l` is linearly scheduled from 1.0
    # at the first block to `1 - drop_path_max` at the last
    # (`p_l = 1 - drop_path_max * l / (n_layers - 1)`, l = 0-indexed layer
    # position). The coin is shared across the batch (one flip per block per
    # step) — matches the paper and avoids per-token noise that hurts causal
    # LM. Eval has no stochasticity: full block, no rescale. Default off →
    # baseline path bit-identical; flag on + drop_path_max=0.1 is the original
    # paper default (ViT-B/16 12L used 0.1; ConvNeXt 18-36L uses 0.1-0.4).
    # See `autoresearch/ideas/111-drop-path/idea.md`.
    use_drop_path: bool = False
    drop_path_max: float = 0.1
    # 131 — LayerDrop (Fan, Grave, Joulin 2019, arXiv:1904.09728, ICLR
    # 2020). Whole-layer stochastic depth: per-block Bernoulli gate
    # during training: with probability `1 - p_l` skip the entire block
    # (`x ← x`); with probability `p_l` keep and rescale by `1/p_l` so
    # the expected residual matches baseline. Coin is shared across
    # the batch (one flip per block per step) — different from
    # DropPath (111) which is per-batch coin AND per-sample (well, here
    # both are per-batch, but LayerDrop is BLOCK-level, not residual-
    # branch-level). `layerdrop_schedule`:
    #   "constant"          → p_l = layerdrop_p for all l (paper default)
    #   "linear"            → p_l = (l/(L-1)) · layerdrop_p (paper stable variant)
    #   "stochastic_depth"  → p_l = layerdrop_p · (l/(L-1)) (drops start at 0)
    # Eval has no stochasticity. With `use_layerdrop=False` (default)
    # the gate is never applied → baseline path bit-identical.
    # NOTE: with the flag ON, step-0 is NOT byte-identical to baseline
    # — the kept-block rescale `1/p_l` magnifies the residual by 1/p_l
    # (e.g. 5× at p_l=0.2). The lever is explicitly an own-control,
    # not an identity trick. See
    # `autoresearch/ideas/131-layer-drop/idea.md`.
    use_layerdrop: bool = False
    layerdrop_p: float = 0.2
    layerdrop_schedule: str = "constant"
    # 024 — Gated Attention (Qiu et al. 2025, arXiv:2505.06708): per-head
    # *scalar* sigmoid gate on the head output `o_h = A_h V_h`, applied
    # post-AV and pre-merge with the O projection: `o_h ← o_h · 2·σ(W_g·x+b)`.
    # `W_g : nn.Linear(d_model, n_heads)` (one scalar gate per head —
    # NOT the per-head vector form, which would blow the parameter budget
    # at this tier). Gate input is the **sublayer input residual** (pre-LN,
    # NOT `o_h` itself — that would be circular). Identity-init: W=0, b=0
    # → 2·σ(0) = 1.0 exactly at step 0, so the pre-norm baseline path
    # stays bit-identical when the flag is off AND when the flag is on
    # at step 0. Categorically distinct from the pre-existing
    # `use_attn_output_gate` (which is a per-head *learnable scalar gain*
    # `o_h *= (1 + g_h)`, not input-conditional, ReZero-style). Default
    # off → baseline path bit-identical. See
    # `autoresearch/ideas/024-gated-attention/plan.md`.
    use_gated_attn: bool = False

    # 147 — DropKey (Xu et al. 2022, arXiv:2207.01058). Per-head
    # Bernoulli gate on K during training: sample `M ~ Bernoulli(1-p)`
    # of shape `[B, n_heads, T, 1]` and apply `K ← K · M / (1-p)`
    # (inverted-dropout rescale so the expected magnitude matches the
    # un-dropped baseline). The mask is per-head, per-token, and
    # independent across batch — finer granularity than DropPath
    # (111, per-batch coin, per-block) and orthogonal to value-side
    # regularizers (use_value_channel_gate, use_kda_channel_gate) and
    # to score-side regularizers (use_fox, use_ssmax). At inference
    # (training mode off) the mask is identity ⇒ forward graph is
    # bit-identical to the no-DropKey baseline. With
    # `use_drop_key=False` (default) the K tensor is never modified,
    # so baseline path is bit-identical at any rate. Default
    # `drop_key_rate=0.1` matches the ViT-B/16 paper default;
    # `drop_key_rate=0.0` collapses to no masking regardless of the
    # flag. See `autoresearch/ideas/147-dropkey/idea.md`.
    use_drop_key: bool = False
    drop_key_rate: float = 0.1

    # 151 — RoV (Rotary Value Embeddings, gated). Apply the same rotary
    # position embedding already used on Q,K to the value vector V as
    # well, mixed in via a learnable per-block scalar gate
    # `rov_gate = nn.Parameter(torch.zeros(1))`. Init 0 ⇒ V_combined =
    # V + 0·V_rot = V ⇒ step-0 forward graph bit-identical to baseline.
    # The base rotary is reused from Q,K (no extra buffer). When
    # `use_nope`/`use_cope` is on, RoPE is bypassed and RoV is a no-op
    # (the geometric lever is unavailable). Default off → baseline
    # path bit-identical. See `autoresearch/ideas/151-rov-gated/idea.md`.
    use_rov: bool = False

    # 156 — Mixture-of-Attentions (MoA). Run `E` parallel attention
    # computations per layer with separate K_e, V_e projections (Q
    # is shared across experts). Mix the E attention outputs by a
    # per-token router `g_e = softmax(W_g x)_e`. At init the
    # (E-1) extra K/V projections are zero (extra experts produce
    # 0 attention) and the router bias is one-hot on expert 0
    # (g_0 = 1.0) ⇒ step-0 output is bit-identical to a single
    # standard attention. Distinct from MoS (144, closed) which
    # mixes softmax variants within one attention — MoA mixes full
    # attention computations. Default off → baseline path bit-
    # identical (no MoA parameters built, no MoA branch taken).
    # `moa_num_experts=E` is the expert count; default E=2 (one
    # extra expert). Cost when on: (E-1) × (2·kv_size × d_model)
    # extra K/V + d_model × E router params per layer ≈ 4-5K
    # params/layer at tiny1m3m (~5% of the 0.94M model). See
    # `autoresearch/ideas/156-moa/idea.md`.
    use_moa: bool = False
    moa_num_experts: int = 2

    # 154 — Rebased Attention (Shi et al. 2024, arXiv:2407.06641):
    # pool K and V along the time axis with a fixed stride-R average
    # *before* the softmax, so attention reads from a learned set of R
    # summary positions instead of T raw ones. `rebase_stride=R` (default
    # 8) gives R = ceil(T/R) ≈ 256 rebasins at tiny1m3m's T=2048.
    # Implementation: `K' = avg_pool(K, R)`, `V' = avg_pool(V, R)`,
    # then `softmax(Q @ K'^T) @ V'` with a causal mask at the
    # rebased-time level (query t can only attend to rebasin r when
    # t >= r·R). When `rebase_stride >= T` the pool collapses to a
    # single block per token, equivalent to the standard full attention
    # → bit-identical to baseline. When `use_rebased_attn=False`
    # (default) the rebase branch is never built and the standard
    # softmax path runs unchanged. Forces the manual attention path
    # (the rebased causal mask can't go through SDPA's flash kernel).
    # See `autoresearch/ideas/154-rebased-attn/idea.md`.
    use_rebased_attn: bool = False
    rebase_stride: int = 8

    # 134 — Mega: Moving Average Equipped Gated Attention
    # (Ma et al. 2022, arXiv:2209.10655, ICLR 2023). Replaces the
    # standard V projection with `V_mega = concat(V, V_ema)` where
    # `V_ema = β·V_ema_{t-1} + (1-β)·V_raw_t` is a learned per-channel
    # exponential moving average over the V projection input. The
    # attention weights then softmax over the doubled key dim and
    # the AV product sums over both halves. `mega_beta` is a
    # per-channel learnable scalar in `[0, 1]` parametrized as
    # `β = σ(raw)` so it stays bounded during training; raw is
    # zero-init ⇒ β = 0.5 at step 0 (the natural "half-smoothed"
    # midpoint between the paper's β=0 "no smoothing" and β=1
    # "constant EMA" extremes). At step 0 β=0.5 ⇒ V_mega is NOT
    # identical to V; the lever is explicitly NOT a baseline-
    # identity trick. At β=0 the EMA collapses to the current
    # token's V (concat → gated attention, the closed 024 lever);
    # at β=1 it collapses to a constant u_t (no signal). The Mega
    # paper's default β=0.9 is achieved when `raw = log(9) ≈ 2.2`;
    # the optimizer finds the right operating point during
    # training. `mega_use_input=True` (default) feeds the residual
    # stream `x` into the EMA; `False` would feed the projected
    # V (but `x` carries the most recent context and is what the
    # paper actually smooths). Default off → baseline path
    # bit-identical (no Parameter created, no concat applied).
    # See `autoresearch/ideas/134-mega-ema/idea.md`.
    use_mega: bool = False
    mega_beta: float = 0.9  # paper default; raw scalar parametrized as sigmoid
    mega_use_input: bool = True  # EMA on pre-projection residual x (paper form)

    # 129 — YOCO: You Only Cache Once (Sun et al. 2024, arXiv:2405.05254,
    # ICLR 2024 workshop). Decoder-decoder cross-layer KV reuse: the
    # model is split into a lower half (standard sliding-window self-
    # attention, default `yoco_lower_window=512`) and an upper half
    # where each layer's attention reads a SHARED `(K_g, V_g)` cache
    # projected from the lower half's final residual stream — instead
    # of computing per-layer K, V from the input. Saves ~50% of the
    # upper-half K/V projection params (the W_K, W_V slices of
    # `qkvo_proj` are unused on the upper half) and at inference
    # collapses the KV cache from `O(L·d·T)` to `O(d·T)`. The lever
    # is the cross-layer information flow itself, not the cache
    # saving (which doesn't affect the tiny1m3m val-loss A/B).
    # `yoco_split` is the 0-indexed layer where the split happens (the
    # LAST lower-half layer is `yoco_split - 1`; the FIRST upper-half
    # layer is `yoco_split`); with default 6 and n_layers=12 the lower
    # half has 6 layers, the upper half has 6 layers. The lower half
    # runs standard sliding-window self-attention (turning
    # `use_sliding_window=True` and `sliding_window_size=yoco_lower_window`
    # on those blocks only). The upper half uses `YOCOLlamaBlock` whose
    # MHA has `use_shared_kv=True` — the K, V projections are skipped
    # and replaced with a single shared `GlobalKVHead` projection that
    # runs ONCE on the lower-half output. Identity at step 0: the
    # `GlobalKVHead` projections have normal init std=0.02 (matching
    # the rest of the model), so K_g, V_g are `O(0.02)` at step 0 →
    # upper-half attention output is small but non-zero. NOT
    # byte-identical to the standard self-attention baseline at
    # step 0 (the standard path uses per-layer K, V projections
    # with std=0.02 init → same magnitude order), but the deviation
    # is bounded by `O(0.02²)` which is within the NULL band. With
    # `use_yoco=False` (default) the YOCO path is never built and
    # the baseline forward graph is bit-identical.
    # See `autoresearch/ideas/129-yoco/idea.md`.
    use_yoco: bool = False
    yoco_split: int = 6
    yoco_lower_window: int = 512

    # 117 — Soft MoE (Puigcerver, Riquelme, Mustafa, Houlsby 2024,
    # arXiv:2406.06589, ICLR 2025). Drop-in FFN replacement: E parallel
    # narrower FFNs + softmax-based dispatch/combine so gradients flow to
    # all experts (no top-k, no balancing loss, no straight-through).
    # Each expert has width `d_ff / soft_moe_n_experts` so total FFN
    # params stay at the budget. Dispatch and combine are derived from
    # small per-token linear projections (`W_d, W_c` of shape
    # `[soft_moe_n_experts * soft_moe_n_slots, d_model]`) — zero-init
    # ⇒ uniform softmaxes at step 0 ⇒ every expert sees roughly the
    # same weighted average of all input tokens ⇒ layer collapses to
    # ~a single FFN applied to `mean(X)`. NOT byte-identical to the
    # single-FFN baseline when flag is ON (the mean-over-tokens
    # aggregation changes the per-token output), but with
    # `use_soft_moe=False` (default) the `SoftMoEFFN` module is never
    # built and the baseline path is bit-identical. See
    # `models/soft_moe.py` for the full mechanism +
    # `autoresearch/ideas/117-soft-moe/idea.md`.
    use_soft_moe: bool = False
    soft_moe_n_experts: int = 4
    soft_moe_n_slots: int = 4

    # 118 — Mixture-of-Depths (Raposo et al. 2024, arXiv:2404.02258):
    # per-token router at each transformer block decides whether the
    # block fires for the token. Top-k tokens (k = mod_capacity·T) get the
    # block's residual update, the rest are passed through unchanged. The
    # kept tokens' residual update is rescaled by `c = k/T` so the
    # expected per-token contribution matches the dense baseline.
    # `mod_capacity=0.5` is the paper's default. `mod_router_hidden=64`
    # gives a 2-layer MLP `W_1 ∈ R^{d×h}`, `W_2 ∈ R^{h×1}` per block;
    # zero-init both ⇒ `σ(0) = 0.5` uniform scores ⇒ top-k is an
    # arbitrary subset at step 0. With `use_mod=False` (default) the
    # `MoDRouter` is never built and the baseline forward graph is
    # bit-identical. See `models/mod_router.py` +
    # `autoresearch/ideas/118-mixture-of-depths/idea.md`.
    use_mod: bool = False
    mod_capacity: float = 0.5
    mod_router_hidden: int = 64

    # 148 — Focal Modulation Networks (Yang et al. 2022,
    # arXiv:2203.11926, NeurIPS 2022). Replaces the attention sub-block
    # with a three-stage focal modulator: (1) hierarchical context
    # aggregation via a stack of depthwise causal Conv1d at multiple
    # kernel sizes (default 3, 5, 7); (2) gather linear that projects
    # the multi-scale context into modulation space; (3) modulate via
    # `output = x + σ(W_g x + b_g) * (W_q x ⊙ W_h · context)`. Different
    # inductive bias from softmax attention: no QKᵀ, no softmax, no
    # O(T²) memory. Step-0 identity: `gather` and `h_proj` are both
    # zero-init, so the modulation signal is exactly `0` at step 0
    # and `output = x` — bit-identical to baseline when flag is off.
    # With `use_focal_mod=False` (default) the `FocalModulationBlock`
    # is never built and the MHA path is bit-identical. Cost when on:
    # 3 × d_model × K depthwise conv params + 3 × d_model² linear
    # params ≈ 3 × 64 × 5 + 3 × 64² ≈ 960 + 12,288 ≈ 13.2K extra
    # params per block × 12 blocks ≈ 159K (~17% of the tiny1m3m
    # budget). See `autoresearch/ideas/148-focal-mod/idea.md`.
    use_focal_mod: bool = False
    focal_mod_kernels: tuple = (3, 5, 7)

    # 146 — Switch FFN (Fedus, Zoph, Shazeer 2022, arXiv:2101.03961):
    # replace the dense FFN with N parallel FFN "experts" and a
    # top-1 learned router per token. The simplest form of sparse
    # mixture-of-experts in the FFN position. Distinct from
    # 117-soft-moe (slot assignment, all experts always used) and
    # 118-MoD (skip-routing) — Switch uses *top-1 hard routing*.
    # When `use_switch_ffn=True`, swap the standard dense FFN for
    # `SwitchFFN` (E parallel full-width FFNs + top-1 router).
    # Each expert is full-width (no narrowing), so the FFN-param cost
    # multiplies by `n_ffn_experts` (default 4×) — a real param
    # injection. `expert_capacity_factor` controls the per-expert
    # token cap = `ceil(N/E) * capacity_factor`; tokens beyond
    # capacity pass through unchanged (residual identity, paper §2.2).
    # Identity at step 0: `W_router` is zero-init ⇒ argmax over
    # uniform-zero returns index 0 for every token ⇒ all tokens
    # route to expert 0 ⇒ output = expert_0(x) = a standard dense
    # FFN (with the same squared_relu/swiglu/etc. variant the
    # baseline would have used). With `use_switch_ffn=False`
    # (default) the `SwitchFFN` module is never built and the
    # baseline FFN path is bit-identical. See `models/switch_ffn.py`
    # and `autoresearch/ideas/146-sparse-ffn/idea.md`.
    use_switch_ffn: bool = False
    n_ffn_experts: int = 4
    expert_capacity_factor: float = 1.25

    # 145 — Expert-Choice MoE (Zhou, Lei, et al. 2022,
    # arXiv:2202.09368). Inverted routing direction vs Switch FFN:
    # each expert picks its own top-k tokens (k = ceil(N/E)) instead
    # of each token picking its top-1 expert. Load balance is by
    # construction — every expert processes exactly k tokens — so
    # NO auxiliary load-balancing loss is required. When
    # `use_expert_choice_moe=True`, swap the standard dense FFN for
    # `ExpertChoiceMoE` (E parallel full-width FFNs + a
    # `nn.Linear(d_model, n_experts)` zero-init router). Each expert
    # is full-width so the FFN-param cost multiplies by `n_moe_experts`
    # (default 4×). At step 0 the router is zero-init ⇒ all
    # expert-token scores are 0 ⇒ every expert processes the same
    # set of k tokens with uniform softmax weights ⇒ output ≈
    # uniform mean of E identically-init'd FFNs (close to a single
    # FFN but NOT byte-identical — same caveat as 117-soft-moe).
    # With `use_expert_choice_moe=False` (default) the
    # `ExpertChoiceMoE` module is never built and the baseline FFN
    # path is bit-identical. See `models/expert_choice_moe.py` and
    # `autoresearch/ideas/145-expert-choice/idea.md`.
    use_expert_choice_moe: bool = False
    n_moe_experts: int = 4

    # 149 — TTT-Linear (Sun, Yang, et al. 2024, arXiv:2407.04620,
    # §3.2). Drop-in FFN replacement: the FFN's up-projection is
    # swapped for `TTTLinear` — a per-input closed-form fast-weight
    # linear that updates its own weight from the input on the fly
    # (one Newton-style gradient step on the auto-encoding loss
    # `||W·x − x||²`). The down-projection stays a standard
    # `nn.Linear` so the FFN output side is unchanged. Per-input
    # fast weights act as a capacity multiplier: a 0.94M model with
    # per-input W_f behaves like a much larger static model in
    # expectation. The fast path costs O(B·T·out·in) extra FLOPs per
    # layer. `ttt_lr_init=0.0` (default) zero-inits the per-layer TTT
    # learning rate so `lr=0` at step 0 ⇒ `TTTLinear` short-circuits
    # to `F.linear(x, weight, b)` with the same `kaiming_uniform_`
    # weight as `nn.Linear` ⇒ the FFN is bit-identical to a vanilla
    # `SquaredReLUFeedForward` at step 0. With `use_ttt_ffn=False`
    # (default) the `TTTFeedForward` module is never built and the
    # baseline FFN path is bit-identical. See
    # `models/ttt_linear.py` and
    # `autoresearch/ideas/149-ttt-linear/idea.md`.
    use_ttt_ffn: bool = False
    ttt_lr_init: float = 0.0

    # 109 — KDA channel gate (Kimi Linear, arXiv:2510.26692): per-channel
    # *bounded* diagonal gate on the V stream of each head. KDA replaces
    # the single scalar forget/decay gate in delta-rule attention with a
    # per-channel diagonal `Γ = diag(γ_1, …, γ_d)`. In this repo's softmax
    # attention, the closest analog is a per-(head, channel) gate on V
    # before the AV product. Parametrized as a *bounded* `2·σ(g)` (not the
    # unbounded `1+g` of the closed `use_value_channel_gate`) so each
    # channel can independently amplify or dampen its own value stream
    # within `(0, 2)`. `g ∈ R^{n_heads × d_k}` zero-init ⇒ `2·σ(0) = 1.0`
    # exactly at step 0 ⇒ baseline graph bit-identical when the flag is
    # off AND when the flag is on at step 0. Categorically distinct from
    # the closed `use_value_channel_gate` (unbounded, can drift to
    # extremes) and from every active attention-side lever (021-V-residual
    # is cross-layer V, 022-softpick is the softmax swap, 024-gated-attn
    # is post-AV o_h gate, 020-FoX is post-softmax A·D). The lever is the
    # *diagonal* and *bounded* per-channel V gain. Default off → baseline
    # path bit-identical (no Parameter created, no application site taken).
    # See `autoresearch/ideas/109-kda-channel-gate/idea.md`.
    use_kda_channel_gate: bool = False

    # ============================================================================
    # Query-tweaks plan (29 experiments, 6 batches). All defaults are
    # identity/zero-init so step-0 == baseline unless the flag is on.
    # See docs/research-plans/query-tweaks/plan.md for the spec.
    # ============================================================================

    # q-side normalization (Q-only). Defaults to qk_norm_type at
    # construction (see __post_init__) so existing configs are
    # bit-identical. Sweep in Batch 4 (Q11-Q16) sets this directly.
    q_norm_type: str = ""
    # ---- Batch 1: high-signal levers ----
    # Q1 ALiBi-style per-head distance bias `scores += -m_h·(i-j)`.
    use_alibi_bias: bool = False
    # Q2 Token-conditioned per-head temperature `Q *= (1+tanh(x·w_h))`.
    use_q_temp_token: bool = False
    # Q3 Cosine attention (L2-normalize Q,K; learnable per-head τ).
    use_cosine_attn: bool = False
    # Q4 Per-channel relevance `score = Q^T diag(d_h) K` (d_h init 1).
    use_qk_bilinear: bool = False
    # ---- Batch 2: flagship + positional ----
    # Q5 Talking-heads on Q: logit-mix via learned n_h × n_h M (M=I init).
    use_talking_heads_q: bool = False
    # Q6 Per-head learnable RoPE base (Q and K share head h's θ).
    use_per_head_rope_base: bool = False
    # Q7 Partial rotary: rotate only fraction p of Q/K dims (default 1.0).
    partial_rotary_p: float = 1.0
    # ---- Batch 3: exotic ----
    # Q8 Multi-query expansion: project Q to 2·q_size, 2 attention reads, mean.
    use_q_expansion: bool = False
    # Q9 Decoupled content/position attention (DeBERTa-style).
    use_decoupled_content_pos: bool = False
    # Q10 Antisymmetric Q·K coupling via learnable skew S (init 0).
    use_antisym_qk: bool = False
    # ---- Batch 5: learnable-param zoo ----
    # Q17 Per-head bias `Q += b_h` after q_norm and RoPE (constant prior).
    use_q_per_head_bias: bool = False
    # Q18 Per-channel gain `Q *= g_d` (d_k) after RoPE.
    use_q_per_channel_gain: bool = False
    # Q19 Head×channel gain `Q *= g_hd` (n_h × d_k) after RoPE.
    use_q_hd_gain: bool = False
    # Q20 Norm-gate: per-head scalar `g_h = σ(a_h·‖x‖+b_h)` on Q.
    use_q_norm_gate: bool = False
    # Q21 Low-rank refine: `Q ← Q + (W1·x) @ W2` (zero-init, default r=8).
    use_q_lowrank_refine: bool = False
    q_lowrank_refine_rank: int = 8
    # Q22 LayerScale on Q: `Q *= (1 + ls_d)` per-channel after RoPE.
    use_q_layerscale: bool = False
    # Q23 Softplus gain: `Q *= softplus(g_h)` per-head — always ≥ 0.
    use_q_softplus_gain: bool = False
    # ---- Batch 6: architecture / mixing ----
    # Q24 Head-mix: `Q ← Q + Q @ M` (M=I init) pre-attention.
    use_q_head_mix: bool = False
    # Q25 Time-conv: `Q += conv1d(Q, k=3)` zero-init along position axis.
    use_q_time_conv: bool = False
    # Q26 EMA-smooth over position: `Q ← α·Q + (1−α)·Q_prev` (α=1 init).
    use_q_ema_smooth: bool = False
    q_ema_alpha: float = 0.0  # sigmoid'd; 0 → α=0.5 at init
    # Q27 Feature-map attention: phi(Q) @ phi(K)^T with learnable phi.
    # NOT identity-init — see plan.md note. Needs own control.
    use_q_feature_map: bool = False
    q_feature_map_hidden: int = 64
    # Q28 Per-token RoPE: each token's θ via small MLP (default hidden=32).
    use_q_per_token_rope: bool = False
    q_per_token_rope_hidden: int = 32
    # Q29 Noise reg: `Q += N(0, σ²)` in training only (learnable σ).
    use_q_noise_reg: bool = False

    # Base Training Defaults
    seed: int = 42  # seeds model init AND data order; override via --seed
    device: str = "auto"  # auto, cuda, mps, or cpu
    compile_model: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    train_tokens: int = 8000000
    
    # Learning Rate (Aggressive for pre-training)
    muon_lr: float = 0.024
    muon_momentum: float = 0.95
    adamw_lr: float = 0.006
    # SWAN (Ma et al. 2024/2025, arXiv:2412.13148): stateless whitening
    # on matrix gradients. Reuses Muon's lr for the matrix slot; the
    # algorithm has no momentum buffer.
    use_swan: bool = False
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"
    # 112 — Lookahead Optimizer Wrapper (Zhang et al. 2019, arXiv:1907.08610).
    # Wraps the *list* of inner optimizers (Muon, AdamW, ...): every k inner
    # steps, pull slow weights halfway toward fast weights and reset fast to
    # slow. Also clears the inner optimizer's momentum buffers so the next
    # inner step doesn't see stale gradients from before the slow reset.
    # k=5, alpha=0.5 are the paper's defaults. With use_lookahead=False
    # (default) the wrapper is fully inert → baseline path bit-identical.
    # Identity at step 0: slow = theta_init, first inner step uses the
    # baseline Muon/AdamW path unchanged.
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    # 116 — Hyper-Connections (mHC, Xie et al. 2024, arXiv:2409.19606):
    # multi-stream residual that splits `d_model` into `hc_n_resid`
    # parallel streams of width `d_l = d_model // hc_n_resid`, each mixed
    # via per-position `(A_l, B_l, C_l) ∈ R^{n_resid × n_resid}`. Identity
    # init (A=B=C=I) ⇒ streams don't mix at step 0 ⇒ baseline forward
    # graph is bit-identical to the pre-norm residual path (B=A=C=I
    # reduces to `block(x)`). Default off → baseline path bit-identical.
    # Cost: 3·n_resid² × n_layers scalars (288 at tiny1m3m with n_resid=4,
    # negligible). See `autoresearch/ideas/116-hyper-connections/idea.md`.
    use_hyper_connections: bool = False
    hc_n_resid: int = 4
    # 150 — Cross-Layer Feedback Attention (Holtzman et al. 2020,
    # Feedback Transformer, arXiv:2002.09402; lean "previous K=2 layers"
    # variant). Each block reads from a small cache of the previous K
    # blocks' pre-FFN residual states via a `XLayerCrossAttn` head, and
    # adds the result as a gated residual branch. Per-block scalar
    # `xlayer_gate = nn.Parameter(torch.zeros(1))` ⇒ contribution is
    # exactly 0 at step 0 ⇒ baseline forward is bit-identical. K=2 by
    # default (the spec pin — the spec also lets K=4 / 8 be tested).
    # The cross-attn head is single-head with `head_dim=16` to keep
    # params compact at 0.94M: per-block param overhead is
    # 2·d_model·16 + 2·d_model² = 8.2K at d_model=64, ≈10% of the
    # 0.94M budget across 12 blocks. With `use_xlayer_feedback=False`
    # (default) the cross-attn module is never built, the per-block
    # gate is not allocated, and the baseline path is bit-identical.
    # See `models/xlayer_attn.py` and
    # `autoresearch/ideas/150-xlayer-feedback/idea.md`.
    use_xlayer_feedback: bool = False
    xlayer_k: int = 2
    # 115 — R-Drop: Regularized Dropout for Neural Networks
    # (Liang et al. 2021, arXiv:2106.14448, NeurIPS 2021). Run the model
    # forward twice per step with different dropout masks, average the two
    # CE losses, and add `rdrop_alpha · 0.5·(KL(p_1‖p_2)+KL(p_2‖p_1))`
    # to pull the model's logits toward dropout-invariance. `rdrop_alpha`
    # is linearly warmed from 0 → target over `rdrop_warmup_steps` so at
    # step 0 the loss is the (mean of two) CE only — bit-identical to the
    # single-CE baseline modulo the doubled forward (which is runtime,
    # not math). With `use_rdrop=False` (default) the trainer takes the
    # single-forward path → byte-identical to baseline. See
    # `autoresearch/ideas/115-rdrop/idea.md`.
    use_rdrop: bool = False
    rdrop_alpha: float = 1.0   # target alpha; paper sweeps 1.0–5.0
    rdrop_warmup_steps: int = 1000  # step-0 invariance: alpha=0 here
    # 110 — Model-Weight EMA (Polyak-Ruppert averaging, Polyak 1990;
    # used in RoBERTa, MAE, MoCo v3, modded-nanogpt speedrun SWA).
    # Maintain a shadow copy `θ_ema ← μ·θ_ema + (1−μ)·θ` updated each
    # step. `μ` ramps linearly from 0 to `ema_decay` over the first
    # `ema_warmup_steps` ⇒ step-0 EMA = live θ ⇒ step-0 val byte-
    # identical to baseline. `ema_eval_only=True` (default) means the
    # live `θ` is the saved/resumed model and the EMA is *only* swapped
    # in for the val pass; training and checkpointing stay on the live
    # trajectory. With `use_ema_eval=False` (default) the trainer does
    # no shadow copy and the baseline path is bit-identical. See
    # `autoresearch/ideas/110-weight-ema/idea.md`.
    use_ema_eval: bool = False
    ema_decay: float = 0.999
    ema_warmup_steps: int = 100
    ema_eval_only: bool = True
    # 119 — SAM: Sharpness-Aware Minimization
    # (Foret et al. 2020, arXiv:2010.01412, ICLR 2021). Wraps the
    # 1-D / embedding / norm AdamW path with an adversarial ascent
    # step `w ← w + ρ · ∇L(w) / ‖∇L(w)‖` followed by descent at the
    # perturbed point. The Muon 2-D path is unchanged — SAM only
    # applies to the AdamW bucket (per-paper default for Adam-SAM).
    # At step 0 the perturbation is non-zero (O(ρ) along the
    # gradient direction), so the first-step gradient differs from
    # AdamW by O(ρ). With `rho = 0.0` SAM collapses to AdamW (the
    # first_step is a no-op, the second_step is parent's step on
    # the same grad) — the flag-off path stays bit-identical. With
    # `use_sam=False` (default) the trainer uses plain
    # `torch.optim.AdamW` unchanged. See `optimizers/sam.py` for
    # the mechanism and `autoresearch/ideas/119-sam/idea.md` for
    # the bet.
    use_sam: bool = False
    sam_rho: float = 0.05
    # 138 — LookSAM: Periodic Sharpness-Aware Minimization (Du et al.
    # 2022, ICLR 2023, arXiv:2205.13539). Compute-efficient variant of
    # SAM (119): the SAM-style 2-backward ascent-descent step fires
    # only every K steps; the K-1 steps in between are plain AdamW.
    # With paper default K=5, effective compute is ~1.2x (vs. SAM's
    # 2x) at ~80% of the flatness benefit. Mutex with `use_sam`: if
    # both are on, `use_sam` wins (full SAM is the more aggressive
    # variant). With `use_looksam=False` (default) the trainer uses
    # plain `torch.optim.AdamW` unchanged — the LookSAM class is
    # never instantiated, baseline path bit-identical. Identity at
    # step 0: with K=5 the first 4 steps are plain AdamW
    # (`step_count=0..3`, `next_is_sam=False`); the first SAM step
    # fires at `step_count=4`. So LookSAM is *more* bit-identical
    # at step 0 than full SAM (119), which always runs the SAM
    # ascent on the first step. See `optimizers/looksam.py` and
    # `autoresearch/ideas/138-looksam/idea.md`.
    use_looksam: bool = False
    looksam_k: int = 5
    looksam_rho: float = 0.05
    # 121 — Prodigy: An Expeditiously Adaptive Parameter-Free Optimizer
    # (Mishchenko & Defazio 2023, arXiv:2306.06101, NeurIPS 2023 L4DC /
    # COLT 2024). Successor to D-Adaptation (120): smooth *continuous*
    # Adam-style gradient similarity `s_t = ⟨sign(g_t/√v_t), sign(g_{t-k}/√v_{t-k})⟩`
    # feeds `D ← D · exp(β3·s_t)` — eliminating D-Adaptation's noisy
    # binary ramp-up. Plus a *displacement-based* warm-start: the first
    # `prodigy_warmup_steps` (default 10) steps are unit-LR AdamW and
    # `D_0` is set to `‖w_0 − w_k‖ / k` — the natural step size for the
    # measured trajectory, no hand-tuned guess. `prodigy_d0` is the
    # warm-start D scalar (paper default 1.0; *not* the production LR —
    # the production LR is `D_t`, which Prodigy discovers). `beta3` is
    # the D-update coefficient η (paper default 0.01; bounded per-step
    # multiplicative change in [exp(-0.01), exp(0.01)] ≈ [0.99, 1.01]).
    # Identity at step 0: the first `warmup_steps` calls are unit-LR
    # AdamW (i.e. `D_0 = d0` is the multiplier on the AdamW update),
    # so they are NOT bit-identical to AdamW with `adamw_lr` — this is
    # the lever. After warmup, D jumps to the measured displacement
    # and the LR-discovery loop engages. With `use_prodigy=False`
    # (default) the trainer uses `torch.optim.AdamW` unchanged — the
    # Prodigy class is never instantiated. See `optimizers/prodigy.py`
    # and `autoresearch/ideas/121-prodigy/idea.md`.
    use_prodigy: bool = False
    prodigy_d0: float = 0.01
    prodigy_warmup_steps: int = 10
    prodigy_beta3: float = 0.01
    prodigy_d_max: float = 1.0     # paper §3.1 default; upper clamp on D.
                                    # Without this, D grows as e^t per step
                                    # (~1e40 by step 92) and explodes on
                                    # the first small-gradient plateau. The
                                    # re-code uses d0=0.01 *and* d_max=1.0
                                    # for defense in depth (the previous
                                    # d0=1.0 caused a 12.01 → 10348 blowup
                                    # at step 25 of the 2026-06-13 GPU run).
    prodigy_min_d: float = 1e-6    # lower clamp on D (prevents collapse on
                                    # sign-disagreement spike).
    prodigy_update_clip: float = 1.0  # per-param max-norm on
                                       # delta = eff_lr · adam_update. Final
                                       # safety net against a too-large
                                       # eff_lr from a discovery-loop spike.
    # 113 — GaLore: Gradient Low-Rank Projection
    # (Zhao et al. 2024, arXiv:2403.03507, NeurIPS 2024). For each 2-D
    # weight matrix, project the gradient into a rank-`galore_rank`
    # subspace via orthonormal P, Q, run AdamW in the r×r projected
    # space, then project the update back. AdamW state is r×r instead
    # of n×m (memory win, moot at 0.94M). Every `galore_proj_every`
    # steps, P, Q are refreshed from the SVD of a running gradient
    # EMA. Routes ONLY the 2-D non-embed, non-norm slot; 1-D / embed
    # / norm stay on plain AdamW. The forward graph is unchanged, so
    # val_loss at step 0 (computed before any optimizer step) is
    # bit-identical to baseline. The first optimizer step itself
    # differs from AdamW's first step (it operates on a rank-r
    # projection), which is the inherent behavior of GaLore. With
    # `use_galore=False` (default) the trainer's existing Muon path
    # is unchanged. See `autoresearch/ideas/113-galore/idea.md`.
    use_galore: bool = False
    galore_rank: int = 4           # projection rank r (paper sweet spot 4-256)
    galore_proj_every: int = 200   # SVD basis refresh cadence (paper default)
    galore_lr: float = 0.006       # matches adamw_lr; tune in tandem if at all
    galore_beta1: float = 0.9
    galore_beta2: float = 0.999
    galore_eps: float = 1e-8
    # Cautious Muon (Liang et al. 2024, arXiv 2411.16085): one-line sign-mask
    # on the orthogonalized update — zero out components whose sign disagrees
    # with the current gradient. Suppresses stale-momentum artifacts. Bit-
    # identical to baseline when False (default). When True, the masked
    # components reduce effective step norm ~10-20% on average; pair with
    # a small muon_lr bump (e.g. 0.024 → 0.025) to compensate. Applies only
    # to the Muon path; AdamW is unchanged (separate flag `use_cautious_adamw`
    # if/when we add it). See docs/research/muon/cautious-muon/plan.md.
    use_cautious_muon: bool = False
    # Cautious AdamW (Liang et al. 2024, arXiv 2411.16085): same sign-mask
    # as Cautious Muon, applied to the AdamW path (1D / embedding / head).
    # Selects WHICH AdamW bucket(s) the mask fires on (the AdamW path is
    # independent of Muon — `use_cautious_muon` does NOT affect it).
    # Allowed values: "none" (default — bit-identical baseline AdamW),
    # "embedding" (mask on `token_embedding` + `emb_proj` only),
    # "gain" (mask on `*.norm.weight` + 1D scalars), "all" (mask every
    # AdamW param). See autoresearch/ideas/002-cautious-adamw/plan.md.
    use_cautious_adamw: str = "none"
    # SOAP (Vyas et al. 2024, arXiv 2409.11321): Adam in the eigenbasis
    # of the Shampoo preconditioner, with periodic basis refresh. Routes
    # ONLY the 2D non-Muon AdamW params (`token_embedding.weight`,
    # `emb_proj.weight`, `out_proj.weight`) to SOAP; 1D scalars and
    # `*.norm.weight` stay on plain AdamW (eigendecomp is meaningless on
    # 1D). Default off → bit-identical to baseline. Pair with the
    # bf16 pre-flight in `autoresearch/ideas/003-soap/plan.md` before
    # the full screen20m run. See autoresearch/ideas/003-soap/idea.md.
    use_soap: bool = False
    use_soap_precondition_freq: int = 10
    # Schedule-Free AdamW (Defazio et al. 2024, arXiv:2405.15682): eliminates
    # the LR schedule by maintaining a Polyak-Ruppert average alongside the
    # gradient-following iterate. Drop-in replacement for the AdamW path only;
    # Muon path is unchanged. Default off → bit-identical to baseline AdamW.
    # When True, the AdamW optimizer's LR scheduler is set to constant (the
    # averaging handles late-training stabilization). See
    # autoresearch/ideas/006-schedule-free-adamw/plan.md.
    use_schedule_free_adamw: bool = False
    # 120 — D-Adaptation (Defazio 2023, arXiv:2301.11933 / arXiv:2201.11941,
    # ICML 2023). Eliminates the learning-rate knob by maintaining a log-scale
    # running lower bound `D` on the distance from `w_init` to `w_optimal` and
    # deriving the effective LR as `lr_t = D_t / ‖g_t‖`. The 1st/2nd moments
    # of AdamW are retained intact — only the outer LR scaling is replaced.
    # Routes ONLY the 1-D / embedding / norm / head path to `DAdaptAdamW`; the
    # Muon 2-D path is unchanged (D-Adapt is ortho to Muon, lives only on the
    # AdamW bucket). At step 0 `D = 1e-6` warm-start ⇒ `lr_0 ≈ 1e-6 / ‖g_0‖`
    # (essentially zero); after ~10–20 steps `D` reaches a typical AdamW-
    # equivalent value. This first-step ramp-up is the lever's signature, not
    # a bug. Default off → trainer uses plain `torch.optim.AdamW` unchanged,
    # baseline path bit-identical. See `optimizers/dadaptation.py` for the
    # mechanism and `autoresearch/ideas/120-dadaptation/idea.md` for the bet.
    use_dadapt: bool = False
    dadapt_d0_lr: float = 1.0     # η, log-LR update constant (paper default 1.0)
    dadapt_min_lr: float = 0.0    # lower clamp on D (paper default 0.0)
    dadapt_d_max: float = 1.0     # upper clamp on D (paper §3.1, default 1.0).
                                  # Also caps the derived lr_t = D/‖g_t‖.
                                  # Required for stability at tiny1m3m — without
                                  # this `D` grows as e^t per step (~1e40 by
                                  # step 92) and explodes on the first small-
                                  # gradient plateau (val 10.81 → 36.89 → 7e15).
    dadapt_eps: float = 1e-8      # floor for lr_t = D/‖g_t‖ (also Adam eps)
    # 114 — MARS: Variance-Reduced AdamW (Yuan et al. 2024,
    # arXiv:2401.03855). Subclass of AdamW that adds a lag-based
    # variance-reduction correction `g̃_t = g_t + mix_coef *
    # (m_{t-lag} − m_{t-2*lag})` to the *gradient* passed to AdamW.
    # Per-parameter `v` is untouched; only the gradient input is
    # modified. Ring buffer of past `exp_avg` snapshots of length
    # `2*lag` is maintained per param. Identity at step 0: the
    # buffer is empty for the first `2*lag` steps ⇒ correction
    # undefined ⇒ g̃_t = g_t ⇒ bit-identical to plain AdamW. Paper
    # default `lag=10`, `mix_coef=0.5`; `lr_scale=1.0` (paper does
    # not require LR re-tuning). With `use_mars=False` (default)
    # the trainer uses `torch.optim.AdamW` unchanged — the
    # MARSAdamW class is never instantiated. See
    # `autoresearch/ideas/114-mars/idea.md`.
    use_mars: bool = False
    mars_lag: int = 10
    mars_mix_coef: float = 0.5
    mars_lr_scale: float = 1.0
    # RetNet retention kernel (Sun et al. 2023, arXiv 2307.08621):
    # per-head learnable decay γ_h replaces softmax attention with a
    # linear-recurrence kernel. v1 = kernel + synthetic probe only
    # (`models/retention.py` + `tests/test_retention.py`); v2 will
    # wire it into `MultiHeadAttention.forward` as a separate PR.
    # Default off → baseline path bit-identical. See
    # autoresearch/ideas/004-retnet-retention/plan.md.
    use_retention: bool = False
    # Lion optimizer (Chen et al. 2023, arXiv:2302.06675): sign-based
    # optimizer that replaces Muon on the 2-D non-embedding, non-norm
    # routing slot when use_lion=True. Default off → Muon path is
    # bit-identical. `lion_lr=3e-4` matches Chen et al.'s default at
    # much larger scale — do not change without sweeping. `use_lion` and
    # the Muon path are mutually exclusive: enabling Lion routes the
    # Muon 2-D slot to Lion (no parallel Muon instance is created).
    # 1-D / embedding / head stay on AdamW — Lion's fixed-LR sign update
    # is known to diverge on the embedding (Chen et al. 2023 §5). See
    # autoresearch/ideas/011-cautious-lion/plan.md.
    use_lion: bool = False
    lion_lr: float = 3e-4
    lion_beta1: float = 0.9
    lion_beta2: float = 0.98
    # Cautious-Lion (Liang et al. 2024, arXiv:2411.16085) — the Cautious
    # sign-mask generalized to Lion. After computing `update = sign(c)`,
    # zero out components whose sign disagrees with the current gradient
    # and rescale by `1 / mask.mean().clamp(min=0.1)` to keep the effective
    # LR constant. Default off → bare Lion, bit-identical to the
    # use_lion=True baseline. Only takes effect when use_lion=True; the
    # `use_cautious_lion` flag is gated by the trainer so it cannot fire
    # on the AdamW path. See autoresearch/ideas/011-cautious-lion/plan.md.
    use_cautious_lion: bool = False
    # Tiger optimizer (Chen et al. 2024, arXiv:2401.16691): sign-based
    # optimizer with per-parameter magnitude EMA — `update = m / (√v + ε)`
    # where `m` is the gradient EMA (β1=0.9) and `v` is the EMA of |g|
    # (β2=0.999). Distinct from Lion (which has unit-magnitude sign
    # updates); Tiger's per-parameter magnitude EMA gives a tighter LR
    # sensitivity and matches AdamW at ~5-10x lower LR (paper).
    # Replaces Muon on the 2-D non-embedding, non-norm routing slot
    # when `use_tiger=True`. Default off → Muon path is bit-identical.
    # 1-D / embedding / head stay on AdamW — Tiger's sign-stable but
    # magnitude-scaled update can be aggressive on the embedding
    # (paper §5.2 recommends AdamW for embedding). Cold-start with
    # `m_0 = 0`, `v_0 = 0` ⇒ first step is `update = 0/ε = 0` ⇒ no
    # parameter change at step 0 ⇒ byte-identical to baseline at
    # step 0 (no paper warmstart `v_0 = |g_0|`, which would shift
    # the first step to a unit sign step). `tiger_lr=1e-3` matches
    # `adamw_lr / 6` (paper-recommended for tiny models). See
    # `autoresearch/ideas/122-tiger/idea.md`.
    use_tiger: bool = False
    tiger_lr: float = 1e-3
    tiger_beta1: float = 0.9
    tiger_beta2: float = 0.999
    tiger_eps: float = 1e-8
    # 123 — CAME: Confidence-guided Adaptive Memory Efficient
    # Optimization (Luo et al. 2023, arXiv:2307.02085, NeurIPS 2023).
    # AdamW replacement for the 1-D / embedding / norm / head path
    # when `use_came=True`. The update is
    #     m_t = β1·m_{t-1} + (1−β1)·g_t
    #     v_t = β2·v_{t-1} + (1−β2)·g_t²
    #     res_t = (m_t − g_t) / (√v_t + ε)
    #     conf_t = max(res_t, 0) + ε
    #     update = m_t / (√v_t + ε) · conf_t / (|m_t| + ε)
    # — i.e. a confidence-rescaled AdamW where the rescaling
    # down-weights updates when the gradient agrees with the
    # running momentum (residual small) and applies a residual-
    # shaped step when they disagree. Cold-start `m_0 = 0`,
    # `v_0 = 0` ⇒ first-step residual is negative, clipped to
    # 0, confidence = ε, update ≈ 0 ⇒ byte-identical to baseline
    # at step 0. Default off → AdamW path unchanged, baseline
    # bit-identical. See `optimizers/came.py` for the mechanism
    # and `autoresearch/ideas/123-came/idea.md` for the bet.
    use_came: bool = False
    came_lr: float = 0.006
    came_beta1: float = 0.9
    came_beta2: float = 0.999
    came_eps: float = 1e-8
    # Per-element magnitude clip on the raw `update` before the LR
    # scaling in `optimizers/came.py`. Bounds any single step's
    # per-element displacement to `±came_update_clip · lr`. Protects
    # against the `m̂ / ε² ≈ 1e16` blowup when `v̂ ≈ 0` and `m̂` is
    # non-trivial (the 2026-06-13 GPU divergence — val loss 10.81 →
    # 6.79e7 at step 25). Default `10.0` is well above the natural
    # ~1.0 per-element magnitude on a healthy trajectory, so it is
    # effectively inactive on a normal Adam-like regime and only
    # triggers in the runaway-`v̂`-zero blowup case. See
    # `autoresearch/ideas/123-came/idea.md` for the post-mortem.
    came_update_clip: float = 10.0
    # 124 — RAdam: Rectified Adam (Liu et al. 2019, arXiv:1908.03265,
    # ICLR 2020). Replaces the AdamW 1-D / embedding / norm / head
    # path with `RAdam` when `use_radam=True`. The 2-D Muon path is
    # unchanged (RAdam is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME). The update applies
    # a variance-bounded correction `ρ_t` to Adam's bias-corrected
    # step: when the variance of `1/(1−β2^t)` is high (early steps),
    # RAdam falls back to an SGD-only `m̂_t` step (no `v̂_t`); once
    # `ρ_t > 4` (≈ `t > 4/(1−β2)`) it switches to the full Adam-
    # normalized update with the variance-aware `√ρ_t` rescale. This
    # *removes the manual warmup knob* — RAdam auto-detects when the
    # effective LR is safe. At step 0 (t=1) `ρ_1 ≪ 4` ⇒ SGD-fallback
    # path ⇒ `update = (1−β1)·g_0`. NOT bit-identical to AdamW's first
    # step (which uses the full Adam-normalized update), but the
    # magnitude is comparable (O(β1) smaller). This first-step
    # divergence is the lever, not a bug. With `use_radam=False`
    # (default) plain `torch.optim.AdamW` is used — baseline
    # bit-identical. `radam_lr=0.006` matches `adamw_lr` (paper does
    # not require re-tuning). See `optimizers/radam.py` for the
    # mechanism and `autoresearch/ideas/124-radam/idea.md` for the bet.
    use_radam: bool = False
    radam_lr: float = 0.006
    radam_beta1: float = 0.9
    radam_beta2: float = 0.999
    radam_eps: float = 1e-8
    # 126 — AdaShift: Decorrelated Adam via Delayed Gradients
    # (Zhou et al. 2019, arXiv:1810.00143, NeurIPS 2019 workshop).
    # Replaces the AdamW 1-D / embedding / norm / head path with
    # `AdaShift` when `use_adashift=True`. The 2-D Muon path is
    # unchanged (AdaShift is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam). The
    # update uses a *delayed* gradient `g_{t-n}²` for the 2nd
    # moment, decorrelating `v_t` from `m_t` (which both use `g_t`):
    #     m_t = β1·m_{t-1} + (1-β1)·g_t
    #     v_t = β2·v_{t-1} + (1-β2)·g_{t-n}²
    #     update = m̂_t / (√v̂_t + ε)
    # Per-parameter state keeps a queue of past `n` gradients
    # (clones, fp32, length bounded by n). The paper's
    # warm-start `v_0 = g_0²` is used on the first step so
    # `v_1 = β2·g_0²` — NOT bit-identical to AdamW's first step
    # (`v_1 = (1-β2)·g_0²`) but same magnitude order (O(β2)
    # different). The first-step displacement is the lever, not a
    # bug. With `n = 0` AdaShift collapses to AdamW; the
    # `adashift_n = 3` default is the paper's recommended delay.
    # With `use_adashift=False` (default) plain `torch.optim.AdamW`
    # is used — baseline path bit-identical. See
    # `optimizers/adashift.py` for the mechanism and
    # `autoresearch/ideas/126-adashift/idea.md` for the bet.
    use_adashift: bool = False
    adashift_lr: float = 0.006
    adashift_beta1: float = 0.9
    adashift_beta2: float = 0.999
    adashift_eps: float = 1e-8
    adashift_n: int = 3
    # 135 — Adan: Adaptive Nesterov Momentum with N-Step Lookback
    # (Xie et al. 2022, arXiv:2208.06677, TPAMI 2022 / ICLR 2023
    # workshop). Replaces the AdamW 1-D / embedding / norm / head
    # path with `Adan` when `use_adan=True`. The 2-D Muon path is
    # unchanged (Adan is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam,
    # 126-AdaShift, 127-GC, 128-SD). The mechanism (paper Algorithm
    # 1) combines (1) a 1-step first moment, (2) an N-step lookback
    # variance estimate, and (3) a Nesterov-style extrapolated
    # gradient:
    #     g_la = g_t + β_la · (g_t − g_{t−1})
    #     m_t = β1·m + (1−β1)·g_la
    #     v_t = β2·v + (1−β2)·mean(g_{t..t-N+1}²)
    #     update = m_t / (√v_t + ε)         (no bias correction)
    # `adan_n_lookback=4` is the paper's default N. `adan_lookahead_beta=0.5`
    # is the paper's default Nesterov coefficient. At step 0
    # `prev_grad=None` ⇒ lookahead term falls back to `g_0` ⇒
    # `m_1 = (1−β1)·g_0`, `v_1 = (1−β2)·g_0²` (queue length 1) ⇒
    # `update_0 = g_0 / (|g_0| + ε) ≈ sign(g_0)`. NOT bit-identical
    # to AdamW's first step (which uses bias-corrected
    # `m̂/√v̂`), but the magnitudes are similar — the first-step
    # displacement is the lever's signature. The N=4 lookback ramps
    # in over the first 4 steps. With `use_adan=False` (default) the
    # `Adan` class is never instantiated and the trainer uses
    # `torch.optim.AdamW` unchanged. See `optimizers/adan.py` and
    # `autoresearch/ideas/135-adan/idea.md`.
    use_adan: bool = False
    adan_lr: float = 0.006
    adan_beta1: float = 0.9
    adan_beta2: float = 0.999
    adan_eps: float = 1e-8
    adan_lookahead_beta: float = 0.5
    adan_n_lookback: int = 4
    # 140 — Sophia: Scalable Stochastic Second-order Optimizer
    # (Liu, Wang, et al. 2023, arXiv:2305.14342, ICML 2023). Replaces
    # the AdamW 1-D / embedding / norm / head path with `Sophia` when
    # `use_sophia=True`. The 2-D Muon path is unchanged (Sophia is an
    # AdamW replacement, like 114-MARS, 119-SAM, 121-Prodigy, 135-Adan).
    # The mechanism is the diagonal-Hessian-aware update
    #     m_t  = β1·m + (1−β1)·g_t
    #     h_t  = β2·h + (1−β2)·h_hat_t         (h_hat sampled every k)
    #     update = clip(g, ±ρ) / max(h, ε)
    #     θ_t  = θ_{t−1} − lr·(update + λ·θ_{t−1})   (decoupled WD)
    # The diagonal Hessian is sampled via Hutchinson's trace
    # estimator: `u ~ Rademacher(±1)` per parameter, then
    # `h_hat = u · ∇(g·u)`, computed by an extra backward on the
    # scalar `g·u` (one extra backward every `sophia_hessian_freq`
    # steps — paper default 10, so ~1.1× amortized backward cost at
    # 92 update steps). The trainer handles the extra backward; see
    # `training/trainer.py` for the wiring. Defaults match the
    # paper's 125M model (lr=6e-3, β1=0.965, β2=0.99, ρ=0.04) with
    # a per-parameter `update_clip=1.0` safety guard that bounds
    # the cold-start `h_t≈0` amplification to a single AdamW-
    # magnitude step. With `use_sophia=False` (default) the
    # `Sophia` class is never instantiated and the trainer uses
    # plain `torch.optim.AdamW` unchanged — baseline path
    # bit-identical. See `optimizers/sophia.py` for the mechanism
    # and `autoresearch/ideas/140-sophia/idea.md` for the bet.
    use_sophia: bool = False
    sophia_lr: float = 6e-3
    sophia_beta1: float = 0.965
    sophia_beta2: float = 0.99
    sophia_eps: float = 1e-8
    sophia_rho: float = 0.04
    sophia_hessian_freq: int = 10
    sophia_update_clip: float = 1.0
    # 136 — AdaPNM: Adaptive Positive-Negative Momentum
    # (Ding, Zhou, Zhu, Ye, Jiao 2019, arXiv:1906.01520, NeurIPS 2019).
    # Replaces the AdamW 1-D / embedding / norm / head path with
    # `AdaPNM` when `use_adapnm=True`. The 2-D Muon path is
    # unchanged (AdaPNM is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam,
    # 126-AdaShift, 135-Adan, 127-GC, 128-SD). The mechanism
    # maintains TWO parallel momentum buffers — one for the
    # positive part of the gradient `m+_t = β1·m+_{t-1} +
    # (1−β1)·max(g_t, 0)` and one for the negative part
    # `m-_t = β1·m-_{t-1} + (1−β1)·max(-g_t, 0)`. The combined
    # direction `m_t = m+_t − m-_t` is algebraically equal to
    # the standard EMA `β1·m_{t-1} + (1−β1)·g_t` because
    # `max(g, 0) − max(-g, 0) = g` element-wise — the lever's
    # factored-state trick preserves AdamW's update direction
    # while opening the door to future per-side processing.
    # The 2nd moment `v_t = β2·v_{t-1} + (1−β2)·g_t²` is
    # standard Adam-style. Cold-start `m+_0 = m-_0 = v_0 = 0`
    # ⇒ first-step update = `(1−β1)·g_0 / (√((1−β2)·g_0²) + ε)`,
    # approximately equal to AdamW's first step (within an
    # `O(β1)` factor — AdamW applies bias correction `m̂_1 =
    # m_1 / (1−β1)`, AdaPNM does not). With `use_adapnm=False`
    # (default) plain `torch.optim.AdamW` is used — baseline
    # path bit-identical. See `optimizers/adapnm.py` and
    # `autoresearch/ideas/136-adapnm/idea.md`.
    use_adapnm: bool = False
    adapnm_lr: float = 0.006
    adapnm_beta1: float = 0.9
    adapnm_beta2: float = 0.999
    adapnm_eps: float = 1e-8
    # 127 — Gradient Centralization (Yong et al. 2020, arXiv:2004.01461,
    # ICONIP 2020). Pre-step hook that subtracts the mean from each
    # gradient matrix before the AdamW update runs. For 2-D weight
    # `W ∈ R^{n×m}` the mean is taken along `gc_axis` (default 1, the
    # output axis), giving each output neuron zero-mean input
    # gradient. For 4-D conv weights, the mean is taken per-filter
    # over the spatial axes. The transform is `g ← g − mean(g,
    # dim=axis)` — a single linear operator that removes the
    # constant component without changing the variance.
    # Compositional: when `use_gc=True` and no specific AdamW
    # replacement is active, the trainer routes AdamW-eligible params
    # through `GCAdamW` (subclass of `torch.optim.AdamW`). The per-
    # parameter `(m, v)` state is untouched — only the gradient
    # input is centered. The forward graph is unchanged, so `val_loss`
    # at step 0 (computed before any optimizer step) is bit-identical
    # to baseline. The first optimizer step itself differs from
    # AdamW's first step (the centered gradient has zero mean per
    # output neuron, removing the constant component that AdamW's
    # first step otherwise sees) — this is the lever's signature, not
    # a bug. With `use_gc=False` (default) plain `torch.optim.AdamW`
    # is used — baseline path bit-identical. See
    # `optimizers/grad_centralization.py` for the mechanism and
    # `autoresearch/ideas/127-grad-centralization/idea.md` for the bet.
    use_gc: bool = False
    gc_axis: int = 1
    # 125 — PSGD: Preconditioned Stochastic Gradient Descent
    # (Li, Chen, Milenkovic, Giannakis 2024, arXiv:2405.13856,
    # NeurIPS 2024). The most recent (NeurIPS 2024) high-quality
    # optimizer paper with explicit ≥100M-scale LM wins (GPT-2
    # small/medium/large match or beat AdamW at same compute).
    # Replaces Muon on the 2-D non-embedding, non-norm routing slot
    # when `use_psgd=True`. PSGD learns an online preconditioner
    # that whitens the gradient per axis. For 2-D W ∈ R^{n×m}:
    #     P ← P + α · (g g^T / m − I)        (n×n)
    #     Q ← Q + α · (W W^T / n − I)        (m×m)
    #     update = P · g · Q                  (whitened step)
    #     w ← w − lr · (β·m_prev + (1−β)·update)
    # For 1-D params (norms, biases, embeddings): diagonal `D` with
    # `D ← D + α · (g² − 1)` and `update = D · g`. The 1-D / embedding
    # / norm slot stays on AdamW per the paper's default (we keep
    # the same routing as Muon / Lion / Tiger / GaLore). `psgd_alpha`
    # is the preconditioner EMA rate (paper default 1e-3). `psgd_beta`
    # is the momentum coefficient (paper default 0.9). At α=0 PSGD
    # collapses to SGD-with-momentum. At step 0 (P=I, Q=I, m=0) the
    # first update is `I · g · I = g` and the first step is
    # `w ← w − lr · g` (SGD, not AdamW — the lever's first-step
    # signature). With `use_psgd=False` (default) the `PSGD` class
    # is never instantiated and the trainer uses the existing Muon
    # path bit-identically. See `optimizers/psgd.py` for the
    # mechanism and `autoresearch/ideas/125-psgd/idea.md` for the bet.
    use_psgd: bool = False
    psgd_lr: float = 0.01
    psgd_alpha: float = 1e-3
    psgd_beta: float = 0.9
    # 128 — Spectral Decoupling (Yong, Pehlivan, Morariu, Tsang 2022,
    # arXiv:2202.05380, NeurIPS 2022). Replaces the AdamW 1-D /
    # embedding / norm / head path with `SDAdamW` — a thin subclass
    # of `torch.optim.AdamW` that projects each per-param gradient
    # off the weight direction (`g ← g − (⟨g,w⟩/‖w‖²)·w`) before
    # delegating to AdamW's `.step()`. Decoupled WD `λ·w` is
    # unchanged (it acts along w — magnitude shrinkage is preserved).
    # The 2-D Muon path is unchanged (SD is an AdamW replacement, like
    # 119-SAM, 120-DAdapt, 121-Prodigy, 114-MARS, 123-CAME, 124-RAdam,
    # 126-AdaShift, 127-GC). Identity at step 0: with symmetric inits
    # `⟨g_0, w_0⟩` is small but nonzero, so the projection removes
    # an `O(1/n)` component of `g_0`. NOT bit-identical to AdamW's
    # first step (small `O(1/n)` correction), but the deviation is
    # bounded and well within the NULL band. `sd_lambda=1.0` is the
    # paper's full projection. `sd_lambda=0.0` collapses SD to
    # plain AdamW (the projection is inert). With `use_sd=False`
    # (default) plain `torch.optim.AdamW` is used — baseline path
    # bit-identical. See `optimizers/spectral_decoupling.py` and
    # `autoresearch/ideas/128-spectral-decoupling/idea.md`.
    use_sd: bool = False
    sd_lambda: float = 1.0
    # 137 — AdamP: Adam with Projection-Based Update
    # (He, Liu, Mao, Chen, Zhang 2020, arXiv:2006.08217, NeurIPS
    # 2020). Replaces the AdamW 1-D / embedding / norm / head path
    # with `AdamP` when `use_adamp=True`. The 2-D Muon path is
    # unchanged (AdamP is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam,
    # 126-AdaShift, 127-GC, 128-SD). The mechanism projects the
    # Adam update `Δ = m̂/√v̂` onto the orthogonal complement of
    # `w` (removes the component of Δ along w, leaving only the
    # perpendicular component) so the update rotates direction
    # without changing magnitude. The L2 reg is applied as the
    # paper's `λ · ‖w‖ · ŵ` (pure magnitude shrinkage, no
    # rotation). Identity at step 0: for symmetric inits
    # `‖Δ_0 · w_0 / ‖w_0‖²‖` is `O(1/√d)`, so the projection
    # removes a small component of Δ_0 and the first AdamP step
    # ≈ the first AdamW step modulo an `O(1/√d)` correction. With
    # `adamp_lambda=0.0` the projection is fully inert and
    # `AdamP` collapses to plain AdamW — bit-identical baseline.
    # With `use_adamp=False` (default) plain `torch.optim.AdamW`
    # is used — baseline bit-identical. See `optimizers/adamp.py`
    # and `autoresearch/ideas/137-adamp/idea.md`.
    use_adamp: bool = False
    adamp_lr: float = 0.006
    adamp_beta1: float = 0.9
    adamp_beta2: float = 0.999
    adamp_eps: float = 1e-8
    adamp_lambda: float = 1.0  # projection strength (0.0 = inert)
    # 141 — AdaBelief: Adapting Stepsizes by the Belief in Observed
    # Gradients (Zhuang, Liu, Tran, Hoang, Chang, et al. 2020,
    # arXiv:2010.07468, NeurIPS 2020). Replaces the AdamW 1-D /
    # embedding / norm / head path with `AdaBelief` when
    # `use_adabelief=True`. The 2-D Muon path is unchanged
    # (AdaBelief is an AdamW replacement, like 114-MARS, 119-SAM,
    # 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam, 126-AdaShift,
    # 127-GC, 128-SD, 135-Adan, 136-AdaPNM, 137-AdamP). The
    # mechanism replaces AdamW's 2nd moment `v_t = E[g²]` with
    # the *residual* variance `s_t = E[(g_t − m_t)²] + ε`, where
    # `m_t` is the running momentum. Step magnitude is large when
    # the current gradient agrees with the momentum (small
    # residual — we trust the direction) and small when they
    # disagree (large residual). AdamW does the *opposite* — a
    # large `g²` shrinks the step — which is wrong when a large
    # gradient is a *good* direction, not a noisy one. At step 0
    # `m_0 = 0`, `s_0 = ε`; first-step residual is `g_0 −
    # (1−β1)·g_0 = β1·g_0`, so `s_1 = (1−β2)·β1²·g_0² + ε` and
    # `update_0 ≈ g_0 / √(0.081·g_0² + ε) ≈ 3.5·sign(g_0)` — NOT
    # bit-identical to AdamW's first step (AdamW would use
    # `m̂/√v̂ = g_0/|g_0| = sign(g_0)`), but the magnitude is the
    # same order. The first-step displacement is the lever's
    # signature, not a bug. The forward graph is unchanged, so the
    # *pre-step-0 forward* output is bit-identical to baseline.
    # With `use_adabelief=False` (default) plain `torch.optim.AdamW`
    # is used — baseline path bit-identical. See
    # `optimizers/adabelief.py` and
    # `autoresearch/ideas/141-adabelief/idea.md`.
    use_adabelief: bool = False
    adabelief_lr: float = 0.006
    adabelief_beta1: float = 0.9
    adabelief_beta2: float = 0.999
    adabelief_eps: float = 1e-8
    # Moonlight Muon RMS rescale (Kimi / Moonshot AI, arXiv:2502.16982,
    # Feb 2025): replaces the default `shape_aspect` per-tensor scale
    # with `c·sqrt(max(d_in, d_out))` so every 2-D weight has an
    # approximately unit-RMS element-wise update. Geometric calibration
    # of step magnitude across matrix shapes (1:1 attention heads,
    # 1:4 FFN up). Default off → Muon path is bit-identical to the
    # `shape_aspect` baseline. `moonlight_muon_c=0.2` is the paper's
    # tuned single global knob. See
    # autoresearch/ideas/015-moonlight-muon-rms/plan.md.
    use_moonlight_muon: bool = False
    moonlight_muon_c: float = 0.2
    # #16 QK-Norm (Dehghani et al. 2023, ViT-22B, arXiv:2302.05442): apply
    # a `nn.LayerNorm(d_head)` to Q and K along the head-dim axis, before
    # the attention dot product. Bounds the per-head logit
    # `Q·K/√d_head` to `|·| ≤ √d_head` — prevents logit explosion that
    # softens the softmax at depth. Default off → Q/K stay on the existing
    # RMSNorm (qk_norm_type="rmsnorm", the current baseline), so step-0 is
    # bit-identical. Init γ=1, β=0 → identity at step 0. Affects ONLY the
    # Q/K norms (the residual stream norms stay on `norm_type`); the
    # global `use_layernorm` flag is a heavier hammer that flips every
    # norm in the block. See
    # autoresearch/ideas/016-qk-norm/plan.md.
    use_qk_layernorm: bool = False
    # 029 — V-Norm (Wortsman et al. 2023, arXiv:2309.14322): per-head
    # `nn.LayerNorm(d_head)` on V along `d_head` before the AV product,
    # symmetric partner of 016's QK-Norm. Bounds the per-head V vector
    # magnitude so outlier V entries do not dominate the AV aggregation.
    # Separate `nn.LayerNorm(d_head)` module (no weight sharing with
    # q_norm/k_norm from 016). When v_norm_type is also set, the
    # existing v_norm_type wins (explicit > implicit — the closed-#92
    # lever takes precedence). Default off → no v_norm module is built
    # and the baseline path stays bit-identical. See
    # autoresearch/ideas/029-v-norm/plan.md.
    use_v_layernorm: bool = False

    # 132 — Born-Again Networks: Self-Distillation with EMA Teacher
    # (Furlanello, Lipton, Tschiatschek, Prabhudesai, Urbach 2018,
    # arXiv:1805.04770). Maintain a shadow copy of the model
    # `θ_teacher ← (1−β)·θ_teacher + β·θ_student` updated each
    # optimizer step. Add a per-step distillation term
    # `L_distill = α · T² · KL(softmax(teacher/T) ‖ softmax(student/T))`
    # on top of CE. Identity at step 0: the shadow is a clone of the
    # live init, so the teacher forward produces identical logits to
    # the student ⇒ KL = 0 ⇒ loss = CE (byte-identical to baseline
    # at step 0). With `use_born_again=False` (default) the teacher
    # is never built and the loss term is zero ⇒ baseline path
    # bit-identical throughout. See `autoresearch/ideas/132-born-again/idea.md`.
    use_born_again: bool = False
    born_again_beta: float = 0.999  # EMA "speed" (higher = teacher tracks closer)
    born_again_alpha: float = 1.0   # KL weight on top of CE
    born_again_temp: float = 2.0    # distillation temperature; KL scaled by T²

    # 133 — SeqMix: Token-Level Mixup for Language Modeling
    # (Guo, Mao, Zhang 2019, arXiv:1908.02951, extended to LM).
    # When on, the trainer samples a paired sequence from the batch,
    # computes embeddings for both via the model's existing
    # token_embedding (and emb_proj if emb_rank is set), and mixes
    # them at the embedding level:
    #   emb_mixed = λ · emb_a + (1 − λ) · emb_b,   λ ~ Beta(α, α)
    # The residual stream is fed `emb_mixed * emb_scale`; the rest of
    # the model runs unchanged. The loss is the λ-weighted mix of the
    # two CEs against the unmixed targets:
    #   L_mixed = λ · CE(logits, y_a) + (1 − λ) · CE(logits, y_b)
    # α=0.4 is the paper default; λ is almost always in (0.05, 0.95),
    # so the mixed loss differs from the unmixed CE by a non-trivial
    # amount at step 0 — the lever's documented signature. With
    # `use_seqmix=False` (default) `model.seqmix_forward` is never
    # called and the trainer takes the standard `model(x)` +
    # `F.cross_entropy(...)` path — baseline path bit-identical.
    # See `autoresearch/ideas/133-seqmix/idea.md`.
    use_seqmix: bool = False
    seqmix_alpha: float = 0.4

    # Evaluation
    eval_every: Optional[int] = None
    eval_steps: int = 100
    eval_milestones: Optional[Tuple[int, ...]] = None
    
    # Regularization
    weight_decay: float = 0.2
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_amp: bool = True
    ffn_variant: str = "squared_relu"
    
    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        # Query-tweaks Batch 4 prereq wire: q_norm_type defaults to
        # qk_norm_type so existing configs are bit-identical unless the
        # Q-side norm is explicitly set (see plan.md Batch 4 note).
        if not self.q_norm_type:
            self.q_norm_type = self.qk_norm_type

    def active_flags(self) -> dict:
        """Return {field_name: value} for every field whose value differs
        from the LLMConfig default. Used by the metrics writer to dump
        only the non-default knobs — the "what was on" summary of a
        run. Forward-only: new runs emit this; old metrics.json
        don't have it (DESC in runs/make_evidence_index.py is the
        fallback for those).
        """
        import dataclasses
        defaults = LLMConfig()
        out = {}
        for f in dataclasses.fields(self):
            cur = getattr(self, f.name)
            dflt = getattr(defaults, f.name, None)
            if cur != dflt and not f.name.startswith("_"):
                out[f.name] = cur
        return out


# ============================================================================
# SCREEN tier — undertrained (NOT 20x). Cheap, fast filters to find a mechanism's
# sign + basin and kick out bad ideas before paying for a Full run. Screen
# results never transfer-promote; the optimum drifts with training duration.
# ============================================================================


@dataclass
class Screen10M20MConfig(LLMConfig):
    """Screen — ~7.7M params · 20M tokens · ~4880 steps. Confirms sign survives more tokens.

    The 10M architecture: low-rank embedding (emb_rank=48) + depth (24 layers).
    Embedding factorized 49152x144 -> (49152x48)@(48x144), freeing ~4.7M params
    from the lookup table and spending them on transformer depth at a fixed budget.
    """
    d_model: int = 144
    n_heads: int = 6
    n_layers: int = 24
    d_ff: int = 576
    n_kv_heads: int = 2
    emb_rank: int = 48
    max_seq_len: int = 2048
    batch_size: int = 2
    train_tokens: int = 20_000_000
    compile_model: bool = False
    warmup_ratio: float = 0.02
    schedule_type: str = "warmup_decay_to_zero"
    eval_milestones: Optional[Tuple[int, ...]] = tuple(range(0, 4880, 200))


@dataclass
class Screen10M1MConfig(Screen10M20MConfig):
    """Ultra-fast screen — ~10M params · 1M tokens · ~250 steps.

    Kept for checkpoint compatibility and fast experiment screens.
    """
    train_tokens: int = 1_000_000
    eval_milestones: Optional[Tuple[int, ...]] = tuple(range(0, 250, 25))


@dataclass
class Screen10M5MConfig(Screen10M20MConfig):
    """Short screen — ~10M params · 5M tokens.

    Kept for checkpoint compatibility and short transfer checks.
    """
    train_tokens: int = 5_000_000


@dataclass
class Tiny1M3MConfig(LLMConfig):
    """Tiny screen — ~0.94M params · 3M tokens.

    Fast idea filter. This is a separate tier from screen20m:
    use it to rank ideas cheaply, then re-test winners on screen20m
    before making stronger claims.
    """
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 12
    d_ff: int = 256
    n_kv_heads: int = 2
    emb_rank: int = 8
    max_seq_len: int = 2048
    batch_size: int = 2
    train_tokens: int = 3_000_000
    compile_model: bool = False
    warmup_ratio: float = 0.02
    schedule_type: str = "warmup_decay_to_zero"
    eval_milestones: Optional[Tuple[int, ...]] = (
        0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700
    )


@dataclass
class Tiny1M3MReZeroConfig(Tiny1M3MConfig):
    """Tiny1M3M with ReZero residual scaling (Bachlechner et al. 2020,
    arXiv:2003.04887).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Each
    transformer block builds two learnable scalars `α_attn` and
    `α_ffn` (init 0) on the residual adds. At step 0 both gates are
    0, so the residual add becomes a no-op and the model is the
    identity function — bit-identical to baseline in the limit of
    fp32 (the `α·f` term is exactly 0). The optimizer then grows
    the α's during training; the lever is whether layer-specific
    residual scaling helps at 12L. Cost: 2 scalars/block × 12
    blocks = 24 scalars total (negligible).

    Transfer-risk: high. The paper's headline wins are at 100L
    (CIFAR-10 / T2T-ViT) and modest at 12L (GPT-2 125M). tiny1m3m
    is at 12L so the lever is least likely to fire. NULL band
    |Δ| < 0.01. DRIFT > +0.01. PASS ≤ −0.01. See
    `autoresearch/ideas/130-rezero/idea.md`.
    """
    use_re_zero: bool = True


@dataclass
class Tiny1M3MSWANConfig(Tiny1M3MConfig):
    """Tiny1M3M with SWAN on the matrix-weight slot."""
    use_swan: bool = True


@dataclass
class Tiny1M3MShortConvConfig(Tiny1M3MConfig):
    """Tiny1M3M with pre-attention ShortConv (Hyena ShortConv variant).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Adds a depthwise causal Conv1d on the residual stream, applied
    BEFORE the attention sublayer (pre-attention, pre-LN on the conv
    path; right after the canon_conv branch when both are on). The
    conv has identity init (last tap = 1, rest = 0) and is gated by a
    per-block scalar `g` (init 0), so at step 0 `g · ShortConv1D(x) = 0`
    and the block reduces to the no-conv baseline. The conv's
    identity init is a *good* starting structure that the gate can
    smoothly grow into once training begins.

    From Poli, Massaroli, et al. 2023, "Hyena Hierarchy: Towards Larger
    Convolutional Language Models" (arXiv:2302.10866) — specifically
    the ShortConv variant (single depthwise Conv1d, kernel 3 or 4,
    pre-attention local aggregator). The lever is qualitatively
    different from 023-canon-conv (post-attention concat, kaiming
    init) — ShortConv is *pre-attention* and *identity-init*, so the
    conv's local-mixing contribution is built on a "pass-through"
    starting point rather than a learned filter.

    Strictly orthogonal to canon_conv (different placement, different
    init) and to attention-side levers (FIRE / CoPE / FoX / Softpick).
    Default off → baseline path bit-identical. Cost: n_layers ×
    (kernel·d_model + 1) extra params (~2.3K at tiny1m3m, +0.25%).

    NULL band |Δ| ≤ 0.01. DRIFT > +0.01. PASS ≤ −0.01. See
    `autoresearch/ideas/143-shortconv/plan.md`.
    """
    use_short_conv: bool = True
    short_conv_kernel: int = 3


@dataclass
class Tiny1M3MEmbLayerNormConfig(Tiny1M3MConfig):
    """Tiny1M3M with embedding pre-LayerNorm (159).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Adds a
    single `nn.LayerNorm(d_model)` immediately after the scaled
    token embedding, before the transformer stack — the LLaMA 3 /
    Gemma 2 / Mistral / Qwen 2.5 pattern. Different from the closed
    norm zoo (which normalizes *inside* each block); this normalizes
    *once at the input*. The LN is init with
    `weight = std(x_post)`, `bias = mean(x_post)` (empirical, computed
    at construction) so `LN(x_post) ≈ x_post` at step 0 within fp32
    rounding noise — baseline path bit-identical at step 0. Cost:
    2·d_model params (128 at tiny1m3m, ~0.014% of the 0.94M model —
    negligible). See `autoresearch/ideas/159-emb-layernorm/idea.md`.
    """
    use_emb_layernorm: bool = True


@dataclass
class Tiny1M3MReLU2FFNConfig(Tiny1M3MConfig):
    """Tiny1M3M with Squared-ReLU FFN activation (So et al. "Primer",
    arXiv:2109.08668, 2021; Mercury Coder / Inception Labs, 2024).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Replaces the
    FFN activation with `relu2(x) = x * F.relu(x)` (≡ `(max(0, x))^2`)
    via the `use_relu2_ffn` lever — two-projection shape (up_proj,
    down_proj, dropout) so param count matches `SquaredReLUFeedForward`
    and the comparison isolates the activation change. At init with
    normal-distributed pre-activations, both GELU and `ReLU²` produce
    zero-mean, similar-variance outputs; first-forward max-abs-diff
    < 1e-3 in fp32 (well inside the harness tolerance for non-bit-
    identical flags). The branch sits AHEAD of the `ffn_variant`
    cascade in `TransformerBlock` so the activation swap isn't silently
    shadowed by another active FFN-replacement flag.

    Transfer-risk: low. Primer tested at 125M–1.5B (matched SwiGLU on
    language modeling with one fewer matmul); Mercury Coder ships
    `ReLU²` in production. NULL band |Δ| ≤ 0.01. DRIFT > +0.01. PASS
    ≤ −0.01. See `autoresearch/ideas/153-relu2-ffn/idea.md`.
    """
    use_relu2_ffn: bool = True


@dataclass
class Tiny1M3MConvFFNConfig(Tiny1M3MConfig):
    """Tiny1M3M with depthwise Conv inside FFN (ConvBERT/ConvNeXt-style).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4216
    on the Vast V100 box). Adds a symmetric depthwise Conv1d (kernel
    3, identity-init center tap) on the FFN output (post-FFN, pre-
    residual-add). The conv is a strict identity at step 0 ⇒ the
    baseline path is bit-identical when the flag is off. Per-block
    cost: 3 × d_model = 192 params; total at tiny1m3m: 12 × 192 =
    2,304 params (+0.25% of the 0.94M model).

    Different from 143-shortconv (closed null): 143 sits *pre-attention*
    on the residual stream with a CAUSAL conv (last-tap identity init,
    per-block scalar gate init 0); 157 sits *post-FFN* on the FFN
    output with a SYMMETRIC conv (center-tap identity init, no gate).
    Both are local-mixing levers but on different sides of the
    attention sublayer.

    From ConvBERT (Jiang et al. 2020, arXiv:2008.02496) and ConvNeXt
    (Woo et al. 2020) — depthwise conv inside FFN for parameter-
    efficient local mixing. Transfer risk is low (≥100M source scale,
    multiple replications).

    NULL band |Δ| ≤ 0.01. DRIFT > +0.01. PASS ≤ −0.01. See
    `autoresearch/ideas/157-conv-ffn/idea.md`.
    """
    use_conv_ffn: bool = True
    conv_ffn_kernel: int = 3


@dataclass
class Tiny1M3MVMixConvConfig(Tiny1M3MConfig):
    """Tiny1M3M with post-attention V-mix depthwise conv (163).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Each
    block's MHA applies a symmetric depthwise Conv1d to the
    post-attention tensor `[B, T, d_model]` BEFORE the W_O output
    projection (post-SDPA, post-reshape, pre-W_O). Conv weights are
    identity-initialized (center tap = 1, rest = 0) via a raw
    `nn.Parameter(zeros(d_model, 1, k))` with the center-tap set
    inline, so the conv is a strict identity at step 0 ⇒ the
    block's attention output is bit-identical to baseline at step 0
    (within fp32 rounding noise of the conv arithmetic).

    Third axis of a deliberate 3-axis locality test: 143-shortconv
    (pre-attention, closed borderline-WIN-rule), 157-conv-ffn
    (post-FFN-activation, closed null), 163-v-mix-conv
    (post-attention on V, this one). A WIN would localize the
    locality prior to the post-softmax V axis; a NULL closes the
    post-attention locality axis at 0.94M alongside the closed
    pre-attention and post-FFN axes. Both outcomes are informative.

    From Poli et al. "Hyena" (2023, arXiv:2302.10866) — the
    Striped Hyena 7B published architecture uses gated convolutions
    on V before the O projection; the lever is the *residual*
    post-attn V-conv form (not the full Hyena replacement).
    `v_mix_conv_kernel` defaults to 3 (spec pin); valid range is
    odd integers ≥ 3. Default off → baseline path bit-identical.

    @dataclass-decorated so `use_v_mix_conv`/`v_mix_conv_kernel` defaults
    are properly overridden (the dataclass-inheritance pitfall
    documented in `_arq_161-dyt-temp.py`).

    NULL band |Δ| ≤ 0.01. DRIFT > +0.01. PASS ≤ −0.01. See
    `autoresearch/ideas/163-v-mix-conv/idea.md`.
    """
    use_v_mix_conv: bool = True
    v_mix_conv_kernel: int = 3


@dataclass
class Tiny1M5MConfig(Tiny1M3MConfig):
    """Tiny screen — ~0.94M params · 5M tokens.

    Same architecture as Tiny1M3MConfig, longer only when a 3M result
    looks promising but too undertrained.
    """
    train_tokens: int = 5_000_000
    eval_milestones: Optional[Tuple[int, ...]] = (
        0, 50, 100, 150, 200, 300, 400, 500, 600, 750, 900, 1100, 1200
    )


@dataclass
class Tiny1M3MQGainConfig(Tiny1M3MConfig):
    """Tiny1M3M with per-head Q-gain."""
    use_q_gain: bool = True


@dataclass
class Tiny1M3MAttnLogitBiasConfig(Tiny1M3MConfig):
    """Tiny1M3M with per-head attention logit bias (PaLM 2 §arch,
    OLMo 2). A learnable `b_h ∈ R^H` (n_heads=4 at tiny1m3m) is added
    to the attention scores pre-softmax.

    Init `b_h = 0` ⇒ `softmax(scores + 0) = softmax(scores)` byte-for-
    byte at step 0. Forces the manual attention path so SDPA's flash/
    efficient backends don't perturb step-0 numerics.

    NB: mathematically, a *per-head scalar* `b_h` cancels in softmax
    over the key axis for all subsequent steps too (per-(b,h,t)
    `e^{b_h}` factor cancels in the per-row normalizer); the
    experiment is therefore a *recorded null* rather than a useful
    lever. See `autoresearch/ideas/152-attn-logit-bias/idea.md` for
    the full math caveat (PaLM 2's actual `attn_logits` bias is
    `[H, S]`, not `[H]`).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    NULL band |Δ| < 0.005 expected (predicted mathematical null).
    """
    use_attn_logit_bias: bool = True


@dataclass
class Tiny1M3MPerHeadTempConfig(Tiny1M3MConfig):
    """Tiny1M3M with per-head learnable attention temperature
    (PaLM 2 §arch, OLMo 2, Gemma 2). One learnable scalar `τ_h`
    per head (init `1/sqrt(d_k)`), so the per-head logit scale
    becomes `Q_h K_h^T * τ_h` and each head can adjust its own
    sharpness during training.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val
    6.4306). Init `τ_h = 1/sqrt(d_k)` so the lever is byte-
    identical to the baseline `1/sqrt(d_k)` scale at step 0
    (no Parameter perturbation, no branch). Forces the manual
    attention path so SDPA's flash/efficient backends don't
    perturb step-0 numerics. Cost: H scalars/layer (4 at
    tiny1m3m, total 48 — negligible).

    Predictions: most likely a small wash (|Δ| < 0.005); the
    `1/sqrt(d_k)` constant is the canonical default across
    Transformers and the Q/K gradients can absorb the per-head
    scale change. A clear win would be a strong signal that
    the per-head temperature axis was missing; a clear loss
    would suggest a useful prior is being clobbered.

    PASS ≤ ctrl − 0.005. NULL band |Δ| < 0.005. DRIFT > +0.005.
    See `autoresearch/ideas/155-per-head-temp/idea.md`.
    """
    use_per_head_temp: bool = True


@dataclass
class Tiny1M3MPerLayerTempConfig(Tiny1M3MConfig):
    """Tiny1M3M with per-layer learnable attention temperature (161).
    One learnable scalar `τ_l` per layer (init `1/sqrt(d_k)`), so the
    logit scale becomes `Q_h K_h^T * τ_l` — the SAME scale factor
    across all heads in a layer, but different across layers. Each
    layer can adjust its own attention sharpness during training.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val
    6.4306). Init `τ_l = 1/sqrt(d_k)` so the lever is byte-
    identical to the baseline `1/sqrt(d_k)` scale at step 0
    (no Parameter perturbation, no branch). Forces the manual
    attention path so SDPA's flash/efficient backends don't
    perturb step-0 numerics. Cost: 1 scalar/layer (12 at
    tiny1m3m, total 12 — negligible).

    Distinct from `use_per_head_temp` (155): per-head varies
    WITHIN a layer (H scalars/layer, e.g. 4 at tiny1m3m); per-
    layer varies ACROSS layers (1 scalar/layer). The two are
    orthogonal axes — a future composition (per-head × per-
    layer = H·L scalars) would test the full H·L grid.

    Predictions: most likely a small wash (|Δ| < 0.005). The
    `1/sqrt(d_k)` constant is the canonical default across
    Transformers; Q/K gradients can absorb any per-layer scale
    change. A clear win would tell us layers want different
    attention temperatures at this scale (early broad, late
    sharp); a clear null would close the per-layer temperature
    axis and confirm per-block normalization absorbs the
    variance.

    PASS ≤ ctrl − 0.005. NULL band |Δ| < 0.005. DRIFT > +0.005.
    See `autoresearch/ideas/161-dyt-temp/idea.md`.
    """
    use_per_layer_temp: bool = True


@dataclass
class Tiny1M3MHeadGainConfig(Tiny1M3MConfig):
    """Tiny1M3M with per-head RMS gain on the attention output
    (Gemma 2 / Qwen 2.5). After the AV product and softmax
    aggregation, multiply each head's output `o_h = (A·V)_h`
    by a learnable scalar `g_h ∈ R^H`. Init `g_h = 1.0` exactly
    ⇒ `o_h *= 1 = o_h` byte-identical to baseline at step 0.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`,
    val 6.4306). Distinct from qk_norm (016, normalizes the
    *pre*-softmax Q/K magnitudes) — this lever normalizes the
    *post*-AV head-output magnitude instead, so it tests the
    OTHER side of the magnitude axis. Distinct from
    `use_attn_output_gate` (reparam `(1+g_h)` with g_h=0 init):
    that one starts at 1.0 but its magnitude reparam has the
    gradient concentrated in `g_h`; this one is a direct `g_h`
    multiplier so the magnitude *and* gradient are both `g_h`.

    Cost: H scalars/layer (4 at tiny1m3m, total 48 — negligible).

    Prediction: small wash or marginal win (|Δ| < 0.01). The
    post-AV magnitude axis is plausibly a redundant degree of
    freedom given the W_O projection that follows, but heads
    can still learn to attenuate noisy outputs. A clear win
    would extend qk_norm's win on the pre-softmax magnitude
    axis to the post-AV axis; a clear null would close the
    post-AV magnitude axis. A drift (> +0.005) would mean
    the lever is harmful at this scale (over-parameterized
    reinit risk).

    PASS ≤ ctrl − 0.005. NULL band |Δ| < 0.005. DRIFT > +0.005.
    See `autoresearch/ideas/160-rms-gain-per-head/idea.md`.
    """
    use_head_gain: bool = True


@dataclass
class Tiny1M3MYOCOConfig(Tiny1M3MConfig):
    """Tiny1M3M with YOCO: You Only Cache Once
    (Sun et al. 2024, arXiv:2405.05254, ICLR 2024 workshop).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Splits the 12L model at layer 6: the lower half (layers 0..5)
    runs standard sliding-window self-attention with `yoco_lower_window=512`;
    the upper half (layers 6..11) uses `YOCOLlamaBlock` whose attention
    reads a SHARED `(K_g, V_g)` cache projected from the lower half's
    final residual stream. The K_g, V_g projections live on a single
    `GlobalKVHead` module (one `nn.Linear(d_model, kv_size)` per of
    K_g, V_g) and are computed once per forward, before the upper-
    half block loop. The upper-half block's per-layer K, V projections
    are skipped (the MHA's `use_shared_kv=True` flag bails out before
    the W_K, W_V slices of `qkvo_proj`).

    At step 0 the lower half's output is small but non-zero (standard
    init std=0.02 throughout), so K_g, V_g are `O(0.02)` → upper-half
    attention output is `O(0.02²)` per token. NOT byte-identical to
    the standard 12L self-attention baseline at step 0, but the
    magnitude order matches the standard path's per-layer
    `O(0.02²)` step-0 attention output.

    Net new params:
    - `GlobalKVHead`: 2 × d_model × kv_size = 2 × 64 × 32 = 4,096.
    - Upper-half W_K, W_V slices of qkvo_proj: SKIPPED (per-layer
      W_K, W_V still built but unused; they get pruned naturally by
      Muon's orthogonalization). Saves roughly
      `n_upper × 2 × kv_size × d_model = 6 × 2 × 32 × 64 = 24,576`
      params of "active" weight that doesn't drive attention (these
      are still allocated in the merged qkvo_proj, but the gradient
      through W_K_l, W_V_l is zero when use_shared_kv is on). On the
      lower half the full qkvo_proj is used as normal.

    PASS ≤ ctrl − 0.005 (small/null band — at 12L the cross-layer
    KV sharing is plausibly a wash; the win at the original paper's
    32L+ scale is unlikely to transfer to 12L). NULL band |Δ| <
    0.005. DRIFT > +0.005 (the lower-half burden of producing the
    shared K_g, V_g is wasted if the upper half doesn't use them well).
    See `autoresearch/ideas/129-yoco/idea.md`.
    """
    use_yoco: bool = True


@dataclass
class Tiny1M3MCautiousMuonConfig(Tiny1M3MConfig):
    """Tiny1M3M with cautious-muon sign-mask + small LR bump.

    A/B vs the tiny1m ctrl (6.4306) — should land ≤ 6.4206 for a pass.
    """
    use_cautious_muon: bool = True
    muon_lr: float = 0.025  # +4% to compensate for masked components


@dataclass
class Tiny1M3MMoonlightMuonConfig(Tiny1M3MConfig):
    """Tiny1M3M with Moonlight per-tensor RMS rescale on orthogonalized Muon.

    A/B vs the plain-Muon ctrl (`Tiny1M3MConfig`). Replaces the default
    `shape_aspect` per-tensor scale with `c·sqrt(max(d_in, d_out))`
    (Kimi / Moonshot AI, arXiv:2502.16982). c=0.2 is the paper's tuned
    constant — single global knob. PASS ≤ ctrl − 0.01 on val_loss.
    NULL band |Δ| < 0.01. See
    autoresearch/ideas/015-moonlight-muon-rms/plan.md.
    """
    use_moonlight_muon: bool = True


@dataclass
class Tiny1M3MQKNormConfig(Tiny1M3MConfig):
    """Tiny1M3M with QK-Norm (LayerNorm on Q,K before the attention dot product).

    A/B vs the tiny1m ctrl. Replaces the default RMSNorm on Q,K with
    `nn.LayerNorm(d_head)` (γ=1, β=0 init → identity at step 0).
    Residual-stream norms stay on RMSNorm — the lever is strictly the
    per-head logit bounding, not a residual-stream re-centering. PASS
    ≤ ctrl − 0.005 on val_loss (taste review puts leverage at the low
    end of the hypothesis range for 6 layers). NULL band |Δ| < 0.005.
    See autoresearch/ideas/016-qk-norm/plan.md.
    """
    use_qk_layernorm: bool = True


@dataclass
class Tiny1M3MScaleNormConfig(Tiny1M3MConfig):
    """Tiny1M3M with ScaleNorm (scalar-gain RMSNorm on the residual stream)."""
    norm_type: str = "scalenorm"


@dataclass
class Tiny1M3MVNormOnQKNormConfig(Tiny1M3MQKNormConfig):
    """Tiny1M3M with QK-Norm + V-Norm (per-head LayerNorm on V before AV).

    A/B vs the QK-Norm ctrl (`Tiny1M3MQKNormConfig`, the 016 WIN
    signature). Adds a per-head `nn.LayerNorm(d_head)` on V along
    `d_head` (γ=1, β=0 init → identity at step 0), the symmetric
    partner of 016's QK-Norm. Independent `v_norm` module (no weight
    sharing with q_norm/k_norm). Bounds per-head V vector magnitude
    so outlier V entries do not dominate the AV aggregation output.

    PASS ≤ ctrl − 0.005 (matches 016's bar — the symmetric-partner
    bet is at the low end of the hypothesis range per the taste
    caveat that Wortsman used V-norm as a diagnostic, not a primary
    lever). NULL band |Δ| < 0.005. DRIFT > +0.005. See
    autoresearch/ideas/029-v-norm/plan.md.
    """
    use_v_layernorm: bool = True


@dataclass
class Tiny1M3MMoonlightMuonQKNormConfig(Tiny1M3MConfig):
    """Tiny1M3M with Moonlight Muon RMS rescale + QK-Norm stacked.

    A/B vs the plain-Muon ctrl (`Tiny1M3MConfig`). Composition of two
    closed-WIN levers that touch entirely separate code paths:

    - `use_moonlight_muon=True` → optimizer-side per-tensor RMS
      rescale `c·sqrt(max(d_in, d_out))` on the Newton–Schulz
      orthogonalized Muon update (c=0.2, Kimi/Moonshot AI,
      arXiv:2502.16982). Lives in `optimizers/muon.py` / wired
      from `training/trainer.py`. Closed-#015 evidence Δ −0.0138.
    - `use_qk_layernorm=True` → per-head `nn.LayerNorm(d_head)` on
      Q,K head-dim before the dot product (γ=1, β=0 init → identity
      at step 0). Lives in `models/layers.py` MHA forward. Bounds
      runtime `|Q·K/√d_head| ≤ √d_head`. Closed-#016 evidence
      Δ −0.0138.

    No shared state. The composition is a two-flag enable; both
    flag paths are already wired and validated by their parent
    A/Bs. Orthogonality test: additive (~−0.028) → independent
    levers (carry both into the 10M→135M ladder); subadditive
    (~−0.01 to −0.02) → partial overlap, carry the cheaper
    (QK-Norm); null (|Δ|<0.01) → substitutes (carry one).
    A clean null is informative, not a failure. PASS ≤ ctrl − 0.01
    (matches 015's bar). NULL band |Δ| < 0.01. DRIFT > +0.01.
    See autoresearch/ideas/027-moonlight-x-qknorm/plan.md.
    """
    use_moonlight_muon: bool = True
    use_qk_layernorm: bool = True


@dataclass
class Tiny1M3MLionConfig(Tiny1M3MConfig):
    """Tiny1M3M with bare-Lion (Chen et al. 2023) on the 2-D non-embed slot.

    Required prerequisite ctrl for the Cautious-Lion idea — the A/B
    measures `(Cautious-Lion - bare-Lion)`, not `(Cautious-Lion -
    Muon-AdamW)`. Δ vs Muon-AdamW is logged for context but is not the
    pass criterion. `lion_lr=3e-4` is Chen et al.'s default at much
    larger scale; keep it pinned. See
    autoresearch/ideas/011-cautious-lion/plan.md.
    """
    use_lion: bool = True


@dataclass
class Tiny1M3MCautiousLionConfig(Tiny1M3MLionConfig):
    """Tiny1M3M with Cautious-Lion (Liang et al. 2024 sign-mask on Lion).

    A/B vs the bare-Lion ctrl (`Tiny1M3MLionConfig`).
    PASS ≤ −0.015 vs bare-Lion ctrl. NULL band |Δ| < 0.01. DRIFT > +0.01.
    Δ vs Muon-AdamW is a secondary number for context only. See
    autoresearch/ideas/011-cautious-lion/plan.md.
    """
    use_cautious_lion: bool = True


@dataclass
class Tiny1M3MTigerConfig(Tiny1M3MConfig):
    """Tiny1M3M with Tiger (Chen et al. 2024) on the 2-D non-embed slot.

    A/B vs the plain-Muon ctrl (`Tiny1M3MConfig`, val 6.4306). Tiger
    replaces Muon on the 2-D non-embedding, non-norm routing slot —
    the same routing slot that Lion uses (closed 011). 1-D / embedding
    / head stay on AdamW. Tiger's update is `m / (√v + ε)` where
    `m` is the gradient EMA (β1=0.9) and `v` is the EMA of |g|
    (β2=0.999). The per-parameter magnitude EMA is the key
    differentiator from Lion's pure sign update: Tiger's update
    magnitude scales with the recent per-parameter gradient size,
    not unit magnitude. `tiger_lr=1e-3` ≈ `adamw_lr / 6` per paper.
    Forward graph unchanged ⇒ step-0 val_loss is bit-identical to
    baseline (the optimizer only changes step ≥ 1).

    PASS ≤ −0.01 vs the tiny1m3m ctrl. NULL band |Δ| < 0.01.
    DRIFT > +0.01 (Tiger's magnitude scaling is more aggressive
    than Lion's at tiny scale; the paper's LR-5x-lower rule is
    scale-free but the second-moment noise floor differs at 0.94M).
    See `autoresearch/ideas/122-tiger/idea.md`.
    """
    use_tiger: bool = True


@dataclass
class Tiny1M3MCAMEConfig(Tiny1M3MConfig):
    """Tiny1M3M with CAME (Luo et al. 2023) on the AdamW path.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the AdamW 1-D / embedding / norm / head path with
    `CAME` — a confidence-rescaled AdamW variant. The 2-D Muon
    path is unchanged (CAME is an AdamW replacement, not a Muon
    replacement, just like 119-SAM, 120-DAdapt, 121-Prodigy,
    122-MARS, 114-MARS). The update is
        m_t = β1·m_{t-1} + (1−β1)·g_t
        v_t = β2·v_{t-1} + (1−β2)·g_t²
        res_t = (m_t − g_t) / (√v_t + ε)
        conf_t = max(res_t, 0) + ε
        update = m_t / (√v_t + ε) · conf_t / (|m_t| + ε)
    The `conf_t` factor adjusts the update magnitude by the
    momentum/gradient agreement (small residual when they agree
    → tiny update; large residual when they disagree → residual-
    shaped step in the consensus direction). Cold-start `m_0=0`,
    `v_0=0` ⇒ first-step residual is negative, clipped to 0,
    `conf_0 = ε`, update ≈ 0 ⇒ baseline path byte-identical at
    step 0.

    **Round-1 fix (after 2026-06-13 GPU blowup):** `came_lr = 0.001`
    (was 0.006 — paper default, 6× too aggressive at this scale).
    At tiny1m3m the `v̂` noise floor is high (β2=0.999 EMA averages
    over very few samples), and the confidence factor can rescale
    `m̂/denom` beyond ±1 in the residual regime; the paper's `lr =
    adamw_lr` recipe is calibrated for ≥100M where `v̂` averages
    out, not 0.94M / 92-step. The lower LR matches the v1 plan's
    "10-100× scale-down" guidance. Pair with `came_update_clip=10.0`
    in `optimizers/came.py` (per-element magnitude cap on the raw
    `update` to bound the `m̂/ε² ≈ 1e16` blowup in the `v̂≈0`
    cold-start transient).

    PASS ≤ −0.005 (small/null band — the lever's effect is in the
    per-parameter update *magnitude* adjustment on a 92-step
    trajectory, where AdamW's `v` noise floor is already partially
    damped by the long warmup-decay schedule). NULL band |Δ| <
    0.005. DRIFT > +0.005 (the confidence rescaling adds compute
    for no gain at this scale). See
    `autoresearch/ideas/123-came/idea.md`.
    """
    use_came: bool = True
    came_lr: float = 0.001  # 6x lower than paper default (round-1 fix)
    came_beta1: float = 0.9
    came_beta2: float = 0.999
    came_eps: float = 1e-8
    came_update_clip: float = 10.0


@dataclass
class Tiny1M3MRAdamConfig(Tiny1M3MConfig):
    """Tiny1M3M with RAdam: Rectified Adam (Liu et al. 2019, arXiv:1908.03265).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the AdamW 1-D / embedding / norm / head path with
    `RAdam` — the variance-bounded Adam variant. The 2-D Muon path
    is unchanged (RAdam is an AdamW replacement, like 114-MARS,
    119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME). The mechanism:
    compute the closed-form `ρ_t = sma_length / (1−β2^t)` from
    Liu et al. 2019 §3.2. When `ρ_t > 4` (≈ `t > 4/(1−β2)`), RAdam
    uses the full variance-bounded Adam step
    `update = m̂_t · √(ρ_t) / (√v̂_t + ε)`. Otherwise (early steps),
    it falls back to the SGD-only step `update = m̂_t` (no `v̂_t`).
    This auto-detects when the early-step LR spike is unsafe and
    *removes the need for a manual warmup*. The 2-D Muon path is
    unchanged.

    At step 0 (t=1) `ρ_1 ≪ 4` ⇒ SGD-fallback path ⇒ first step is
    `(1−β1)·g_0`. NOT bit-identical to AdamW's first step (which
    uses the full Adam-normalized update), but the magnitude is
    comparable (O(β1) smaller). The first-step divergence is the
    lever's signature, not a bug. With `use_radam=False` (default)
    plain `torch.optim.AdamW` is used — baseline path bit-identical.

    `radam_lr=0.006` matches `adamw_lr` (paper does not require
    re-tuning — RAdam's variance-bounded correction handles the
    early-step instability that warmup usually addresses).

    PASS ≤ ctrl − 0.005 (small/null band — the lever is on the
    early-step LR transition, which only matters in the warmup
    window at tiny1m3m's 92-step budget). NULL band |Δ| < 0.005.
    DRIFT > +0.005 (the SGD-fallback path at step 0 + the
    closed-form correction at later steps adds variance without
    removing it at this scale). See
    `autoresearch/ideas/124-radam/idea.md`.
    """
    use_radam: bool = True
    radam_lr: float = 0.006
    radam_beta1: float = 0.9
    radam_beta2: float = 0.999
    radam_eps: float = 1e-8


@dataclass
class Tiny1M3MPSGDConfig(Tiny1M3MConfig):
    """Tiny1M3M with PSGD: Preconditioned Stochastic Gradient Descent
    (Li et al. 2024, arXiv:2405.13856, NeurIPS 2024).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces Muon on the 2-D non-embedding, non-norm routing slot
    with `PSGD`. 1-D / embedding / norm stay on AdamW (the paper's
    default). PSGD learns an online preconditioner `(P, Q)` for 2-D
    params that whitens the gradient per axis:
        P ← P + α · (g g^T / m − I)         (n×n)
        Q ← Q + α · (W W^T / n − I)         (m×m)
        update = P · g · Q                   (whitened step)
    `psgd_alpha=1e-3` is the paper's recommended preconditioner EMA
    rate. `psgd_lr=0.01` is in the paper's recommended range.
    `psgd_beta=0.9` is the momentum coefficient. The 0.94M context
    is *favorable* to PSGD because the preconditioner matrices are
    small (~4k floats per slot at d_model=64).

    At step 0 (P=I, Q=I, m=0) the first update is `I · g · I = g`
    and the first step is `w ← w − lr · g` (SGD-with-momentum, not
    AdamW — the lever's first-step signature). At α=0 PSGD
    collapses to SGD-with-momentum. With `use_psgd=False` (default)
    the existing Muon path is bit-identical to baseline.

    PASS ≤ ctrl − 0.005 (taste's mid-band for an optimizer-side
    lever at 12L depth; PSGD's per-axis whitening is qualitatively
    different from Muon's orthogonalization, but the lever is
    small at 0.94M where AdamW's per-parameter v is already
    well-conditioned). NULL band |Δ| < 0.005. DRIFT > +0.005 (the
    8k floats of P+Q state per 2-D slot is overhead without
    benefit at this scale). See `autoresearch/ideas/125-psgd/idea.md`.
    """
    use_psgd: bool = True
    psgd_lr: float = 0.01
    psgd_alpha: float = 1e-3
    psgd_beta: float = 0.9


@dataclass
class Tiny1M3MSDConfig(Tiny1M3MConfig):
    """Tiny1M3M with Spectral Decoupling (Yong et al. 2022,
    arXiv:2202.05380, NeurIPS 2022) on the AdamW 1-D / embedding /
    norm path.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val
    6.4306). Replaces the AdamW 1-D / embedding / norm / head
    path with `SDAdamW` — a thin subclass of `torch.optim.AdamW`
    that projects each per-param gradient off the weight
    direction (`g ← g − (⟨g,w⟩/‖w‖²)·w`) before delegating to
    AdamW's `.step()`. The decoupled WD `λ·w` is unchanged (it
    acts along w — magnitude-shrinking role is preserved). The
    2-D Muon path is unchanged (SD is an AdamW replacement, like
    119-SAM, 120-DAdapt, 121-Prodigy, 114-MARS, 123-CAME,
    124-RAdam, 126-AdaShift, 127-GC).

    The forward graph is unchanged, so step-0 `val_loss` (no
    optimizer step yet) is bit-identical to baseline. The first
    optimizer step differs from AdamW's first step by an
    `O(1/n)` correction (the projection removes a small
    `⟨g_0, w_0⟩ / ‖w_0‖² · w_0` term — bounded by the symmetry
    of the init). `sd_lambda=1.0` is the paper's full projection.
    This first-step displacement is the lever's signature, not
    a bug.

    PASS ≤ ctrl − 0.005 (small/null band — the lever's effect
    is on the per-step gradient direction; at 0.94M with the
    warmup-decay schedule the per-step displacement is at most
    `O(1/n)` per param, cumulative over 92 steps ≈ small).
    NULL band |Δ| < 0.005. DRIFT > +0.005 (the projection adds
    compute for no gain at this scale — the rotation effect is
    small when the per-step gradient direction is already
    dominated by m_t/v_t normalization). See
    `autoresearch/ideas/128-spectral-decoupling/idea.md`.
    """
    use_sd: bool = True
    sd_lambda: float = 1.0


@dataclass
class Tiny1M3MAdanConfig(Tiny1M3MConfig):
    """Tiny1M3M with Adan: Adaptive Nesterov Momentum with N-Step Lookback
    (Xie et al. 2022, arXiv:2208.06677, TPAMI 2022 / ICLR 2023 workshop).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the AdamW 1-D / embedding / norm / head path with `Adan`
    — the N-step variance + Nesterov-style extrapolated gradient
    optimizer. The 2-D Muon path is unchanged (Adan is an AdamW
    replacement, like 114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy,
    123-CAME, 124-RAdam, 126-AdaShift, 127-GC, 128-SD). The update
    is (paper Algorithm 1):
        g_la = g_t + β_la · (g_t − g_{t−1})             (Nesterov)
        m_t = β1·m_{t-1} + (1−β1)·g_la
        v_t = β2·v_{t-1} + (1−β2)·mean(g_{t..t-N+1}²)   (N-step)
        update = m_t / (√v_t + ε)                       (no bias-correction)
    `adan_n_lookback=4` is the paper's default N. `adan_lookahead_beta=0.5`
    is the paper's default Nesterov coefficient. The forward graph
    is unchanged, so step-0 `val_loss` (no optimizer step yet) is
    bit-identical to baseline. The first optimizer step has
    `update_0 ≈ sign(g_0)` (no bias correction, queue length 1) —
    NOT bit-identical to AdamW's first step (which uses bias-
    corrected `m̂/√v̂`), but the magnitude is similar and the N=4
    lookback ramps in over the first 4 steps. This first-step
    displacement is the lever's signature, not a bug. With
    `use_adan=False` (default) plain `torch.optim.AdamW` is used —
    baseline path bit-identical.

    PASS ≤ ctrl − 0.005 (small/null band — the lever is on the
    variance *smoothing* over N=4 steps, which has a small effect at
    0.94M where the per-step gradient is already well-behaved).
    NULL band |Δ| < 0.005. DRIFT > +0.005 (the N=4 lookback queue
    is overhead without gain at this scale). See
    `autoresearch/ideas/135-adan/idea.md`.
    """
    use_adan: bool = True
    adan_lr: float = 0.006
    adan_beta1: float = 0.9
    adan_beta2: float = 0.999
    adan_eps: float = 1e-8
    adan_lookahead_beta: float = 0.5
    adan_n_lookback: int = 4


@dataclass
class Tiny1M3MSophiaConfig(Tiny1M3MConfig):
    """Tiny1M3M with Sophia: Scalable Stochastic Second-order Optimizer
    (Liu, Wang, et al. 2023, arXiv:2305.14342, ICML 2023).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4216).
    Replaces the AdamW 1-D / embedding / norm / head path with
    `Sophia`. The 2-D Muon path is unchanged (Sophia is an AdamW
    replacement, like 114-MARS, 119-SAM, 121-Prodigy, 135-Adan).
    The update is the diagonal-Hessian-aware preconditioned step
        m_t  = β1·m + (1−β1)·g_t
        h_t  = β2·h + (1−β2)·h_hat_t       (h_hat every k=10 steps)
        update = clip(g, ±ρ) / max(h, ε)
        θ    ← θ − lr·(update + λ·θ)        (decoupled WD)
    The diagonal Hessian `h_hat` is sampled via Hutchinson's
    trace estimator: `u ~ Rademacher(±1)` per parameter, then
    `h_hat = u · ∇(g·u)` (one extra backward on the scalar `g·u`,
    amortized ~1.1× backward cost at k=10 and 92 update steps).
    The trainer handles the extra backward; see
    `training/trainer.py` for the wiring and
    `autoresearch/ideas/140-sophia/idea.md` for the bet.

    Defaults match the paper's 125M model: lr=6e-3, β1=0.965,
    β2=0.99, ρ=0.04, k=10. `sophia_update_clip=1.0` is a per-
    parameter magnitude safety guard against the cold-start
    `h_t≈0` amplification (the paper does not specify this; we
    add it to bound the very first step to AdamW magnitude). At
    step 0 `m_0 = h_0 = 0` and the first update magnitude is
    bounded by `lr · 1.0`; the first Hutchinson sample fires at
    step `k-1` and `h_t` becomes `O(g²)` thereafter ⇒ proper
    curvature-preconditioned steps from step k onward. This
    first-step guard is the lever's signature, not a bug.

    With `use_sophia=False` (default) plain `torch.optim.AdamW` is
    used — baseline path bit-identical. The Hutchinson extra
    backward is also gated by `use_sophia=True` so the baseline
    training cost is unchanged when the flag is off.

    PASS ≤ ctrl − 0.01 (curvature-aware steps at sub-million
    params have noisy Hessian estimates; gain plausibly shrinks at
    0.94M where the diagonal Hessian has high variance per-step).
    NULL band |Δ| < 0.01. DRIFT > +0.01 (Hutchinson noise at small
    scale hurts more than curvature helps). See
    `autoresearch/ideas/140-sophia/idea.md`.
    """
    use_sophia: bool = True
    sophia_lr: float = 6e-3
    sophia_beta1: float = 0.965
    sophia_beta2: float = 0.99
    sophia_eps: float = 1e-8
    sophia_rho: float = 0.04
    sophia_hessian_freq: int = 10
    sophia_update_clip: float = 1.0


@dataclass
class Tiny1M3MAdaPNMConfig(Tiny1M3MConfig):
    """Tiny1M3M with AdaPNM: Adaptive Positive-Negative Momentum
    (Ding, Zhou, Zhu, Ye, Jiao 2019, arXiv:1906.01520, NeurIPS 2019).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the AdamW 1-D / embedding / norm / head path with
    `AdaPNM`. The 2-D Muon path is unchanged (AdaPNM is an AdamW
    replacement, like 114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy,
    123-CAME, 124-RAdam, 126-AdaShift, 135-Adan, 127-GC, 128-SD).
    The mechanism maintains TWO parallel momentum buffers:
        m+_t = β1·m+_{t-1} + (1−β1)·max(g_t, 0)
        m-_t = β1·m-_{t-1} + (1−β1)·max(-g_t, 0)
        m_t  = m+_t − m-_t
        v_t  = β2·v_{t-1} + (1−β2)·g_t²
        update = m_t / (√v_t + ε)
    The combined direction `m+_t − m-_t` is algebraically equal
    to the standard EMA `β1·m_{t-1} + (1−β1)·g_t` because
    `max(g, 0) − max(-g, 0) = g` element-wise. So today's AdaPNM
    *degenerates to a (no-bias-correction) AdamW* at convergence —
    the dual-momentum buffer is a factored state representation,
    not a different direction. The lever's value is the option
    to apply per-side processing later (e.g. different effective
    β1 for positive vs negative components), not a today's-step
    win.

    Cold-start identity at step 0: `m+_0 = m-_0 = v_0 = 0` ⇒
    first step update = `(1−β1)·g_0 / (√((1−β2)·g_0²) + ε)`.
    NOT bit-identical to AdamW's first step (AdamW applies bias
    correction `m̂_1 = m_1 / (1−β1) = g_0`; AdaPNM does not).
    The first-step displacement is the lever's signature, not a
    bug. The forward graph is unchanged, so step-0 `val_loss`
    (computed before any optimizer step) is bit-identical to
    baseline.

    `adapnm_lr=0.006` matches `adamw_lr` (paper does not require
    re-tuning — the dual-momentum factored state matches AdamW's
    direction up to the bias-correction difference).

    PASS ≤ ctrl − 0.005 (small/null band — the lever is on the
    factored state representation, not the update direction; at
    0.94M with shared `β1` for both halves, today's AdaPNM
    degenerates to a no-bias-correction AdamW). NULL band |Δ| <
    0.005. DRIFT > +0.005 (the 2× memory overhead for `m+` and
    `m-` adds cost without mathematical benefit at this scale
    unless future per-side processing is enabled). See
    `autoresearch/ideas/136-adapnm/idea.md`.
    """
    use_adapnm: bool = True
    adapnm_lr: float = 0.006
    adapnm_beta1: float = 0.9
    adapnm_beta2: float = 0.999
    adapnm_eps: float = 1e-8


@dataclass
class Tiny1M3MBornAgainConfig(Tiny1M3MConfig):
    """Tiny1M3M with Born-Again Networks self-distillation
    (Furlanello et al. 2018, arXiv:1805.04770).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val
    6.4306). Maintains a teacher model as a slow EMA copy of the
    student: `θ_teacher ← (1−β)·θ_teacher + β·θ_student` with
    `born_again_beta=0.999` (paper default). Adds the distillation
    loss `L_distill = α · T² · KL(softmax(teacher/T) ‖ softmax(student/T))`
    with `born_again_alpha=1.0` and `born_again_temp=2.0` on top of
    the standard CE. The teacher's forward uses a parameter-swap
    around `model(x)` under `torch.no_grad()` (no buffers needed —
    dropout is 0.0 by default and RoPE caches are deterministic
    from position ids). One extra forward per step (no_grad ⇒ no
    autograd graph).

    Identity at step 0: the shadow is a clone of the live init ⇒
    teacher forward == student forward ⇒ KL = 0 ⇒ loss == CE. At
    tiny1m3m's 92-step budget, `β=0.999` gives the teacher a
    half-life of `log(0.5)/log(1−0.999) ≈ 693` steps, so the
    teacher is dominated by early-step student states throughout
    the run — the distillation signal is small but persistent.

    PASS ≤ ctrl − 0.005 (small/null band — the lever is on the
    logit-level EMA, not the parameter-level trajectory; at 0.94M
    the EMA is far from saturated and the teacher is close to the
    student most of the time). NULL band |Δ| < 0.005. DRIFT > +0.005
    (the extra forward adds compute without gain at this scale).
    See `autoresearch/ideas/132-born-again/idea.md`.
    """
    use_born_again: bool = True
    born_again_beta: float = 0.999
    born_again_alpha: float = 1.0
    born_again_temp: float = 2.0


@dataclass
class Tiny1M3MGCConfig(Tiny1M3MConfig):
    """Tiny1M3M with Gradient Centralization (Yong et al. 2020,
    arXiv:2004.01461) on the AdamW 1-D / embedding / norm path.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Routes AdamW-eligible params through `GCAdamW`, a thin subclass
    of `torch.optim.AdamW` that subtracts the per-row mean from each
    2-D gradient matrix before the AdamW update runs. For 1-D
    tensors (norms, biases) GC is a no-op (no output-axis semantics);
    for 2-D tensors (the factorized embedding, `emb_proj`, `out_proj`)
    the mean is taken along `gc_axis=1` (the output axis), giving
    each output row zero-mean gradient. The 2-D Muon path is
    unchanged (Muon sees the uncentered gradient and applies its own
    orthogonalization).

    The forward graph is unchanged, so step-0 `val_loss` is
    bit-identical to baseline. The first optimizer step differs from
    AdamW's first step (centered gradient has zero mean per output
    neuron). This first-step displacement is the lever's signature,
    not a bug.

    PASS ≤ ctrl − 0.005 (the lever is on the optimizer input — small
    per-step effect on a 92-step trajectory). NULL band |Δ| < 0.005.
    DRIFT > +0.005 (centering is per-param overhead without benefit
    at this scale if the gradient mean was already near zero). See
    `optimizers/grad_centralization.py` for the mechanism and
    `autoresearch/ideas/127-grad-centralization/idea.md` for the bet.
    """
    use_gc: bool = True
    gc_axis: int = 1


@dataclass
class Tiny1M3MCoPEOnFireConfig(Tiny1M3MConfig):
    """Tiny1M3M with FIRE + CoPE (stacked content-conditional position).

    A/B vs the FIRE-equipped baseline (no `Tiny1M3MFIREConfig` class — the
    FIRE ctrl is just `Tiny1M3MConfig` with `use_fire_pe=True` passed at
    run time; the 009 WIN landed at 6.3234 per `closed.md`). The
    treatment stacks `use_cope=True` on top: CoPE is a *content-
    conditional* positional bias (count of "important" tokens per head,
    Golovneva et al. 2024, arXiv:2405.18719) that REPLACES RoPE and is
    added to attention scores in addition to the FIRE bias. The Q/K
    RMSNorm still runs (it's the magnitude stabilizer, separate from
    position). Probe `p ~ N(0, 0.02)` (mirrors FIRE's per-head content
    init at `models/fire_pe.py:60`); threshold τ pinned at 0
    (one-seed-only rule forbids the τ sweep).

    PASS ≤ −0.01 vs the FIRE-equipped ctrl. NULL band |Δ| < 0.01.
    DRIFT > +0.01. See
    `autoresearch/ideas/013-cope/plan.md`.
    """
    use_fire_pe: bool = True
    use_cope: bool = True


@dataclass
class Tiny1M3MVQGainConfig(Tiny1M3MConfig):
    """Tiny1M3M with V-embed + per-head Q-gain."""
    use_value_embed: bool = True
    use_q_gain: bool = True


@dataclass
class Tiny1M3MSWAConfig(Tiny1M3MConfig):
    """Tiny1M3M with SWA(window=512) only."""
    use_sliding_window: bool = True
    sliding_window_size: int = 512


@dataclass
class Tiny1M3MVQGainSWAHighRoPEConfig(Tiny1M3MConfig):
    """Tiny1M3M with the current screen20m best recipe."""
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000


@dataclass
class Tiny1M3MVQGainHighRoPESWA384Config(Tiny1M3MConfig):
    """Tiny1M3M with V+q+HighRoPE + SWA(window=384)."""
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 384
    rope_base: int = 500000


@dataclass
class Tiny1M3MVQGainSWAHighRoPE250KConfig(Tiny1M3MConfig):
    """Tiny1M3M with V+q+SWA(window=512) + RoPE base 250k."""
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 250000


@dataclass
class Tiny1M3MFOXOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + Forgetting Transformer (multiplicative decay).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val 6.3234
    per `closed.md:40`). Parent is `Tiny1M3MVQGainSWAHighRoPE250KConfig`
    so VQ-gain + SWA(512) + RoPE 250K carry over from the ctrl recipe —
    the A/B isolates the FoX swap (per-head, per-token learnable decay
    on attention probabilities) on top of the same FIRE-equipped
    foundation, not silent HP drift. The treatment stacks `use_fox=True`
    on top: FoX is a *multiplicative* per-head, per-token learnable
    decay on attention *probabilities* (post-softmax), with row-renorm.
    Strictly orthogonal to FIRE (which is *additive* on logits): FIRE
    changes *which* key wins the softmax; FoX changes *how much mass*
    even the winners keep. Conservative extension of softmax attention
    — softmax stays, projection stays, V path is unchanged. b_f = +10
    init → D is within 9% of all-ones over the full T=2048 context at
    step 0, so the model has to *learn* to forget from scratch (gates
    start near 1.0 and can only go down). See `models/fox.py` for the
    identity-init derivation.

    PASS ≤ −0.02 vs the FIRE-equipped ctrl. NULL band |Δ| < 0.02.
    DRIFT > +0.01. See
    `autoresearch/ideas/020-forgetting-attn/plan.md`.
    """
    use_fire_pe: bool = True
    use_fox: bool = True


@dataclass
class Tiny1M3MSoftpickOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + Softpick (rectified-softmax normalization).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:40`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates the
    softpick swap (function-level normalization tweak) on top of the
    same FIRE-equipped foundation, not silent HP drift. The treatment
    replaces `torch.softmax` in the manual attention path with softpick
    `relu(exp(x)−1) / (Σ|exp(x)−1| + ε)`. Permits zero total
    attention mass → kills the attention-sink pathology without
    adding a learnable sink token (categorically distinct from
    the closed `attn-sink` lever, which *added* a sink token;
    distinct from 020-FoX, which multiplies post-softmax; distinct
    from 013-CoPE, which adds a content-aware bias on logits).
    ε=1e-6 pinned; `exp−1` computed in fp32 then cast back.

    PASS ≤ −0.005 vs the FIRE-equipped ctrl. NULL band |Δ| < 0.01.
    DRIFT > +0.01. Step-0 smoke gate (see plan.md) — build trt,
    run one fwd+bwd, assert loss is finite and Q/K/V grads are
    non-zero (zero attn output at step 0 ⇒ zero grad ⇒ lever
    dead on arrival). See
    `autoresearch/ideas/022-softpick-attention/plan.md`.
    """
    use_fire_pe: bool = True
    use_softpick: bool = True


@dataclass
class Tiny1M3MSSMaxConfig(Tiny1M3MConfig):
    """Tiny1M3M with Scalable-Softmax (per-head log(n) attention temperature).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`).
    SSMax multiplies the attention logits by `s_h · log(n)` pre-softmax,
    where n is the per-query causal key count (i.e. n = i+1 at query
    position i) and s_h is a single learnable per-head scalar (init
    1.0). At max_seq_len=2048 the late-position queries attend over
    hundreds-to-thousands of keys, where vanilla softmax provably
    flattens (denominator scales with n, logit variance is fixed).
    SSMax restores per-position sharpness with one scalar per head.
    Distinct from the closed logit-softcap (clamps) and from 020-FoX
    (content decay on probabilities, post-softmax); SSMax is a
    *length-dependent temperature* on logits, an orthogonal axis.
    Stacks on FIRE and on qk-norm cleanly (per-tensor multiplies on
    `scores`; follow-up A/Bs gated on the primary clearing).

    PASS ≤ −0.01 vs the tiny1m3m ctrl. NULL band |Δ| < 0.01.
    DRIFT > +0.01. Anti-cheat: in-bracket ±0.0053 outcomes do not
    count as WIN. See
    `autoresearch/ideas/025-scalable-softmax/plan.md`.
    """
    use_ssmax: bool = True


@dataclass
class Tiny1M3MCanonOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + Canon conv (gated depthwise causal Conv1d).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:40`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates
    the canon-conv swap (one depthwise causal Conv1d per block on
    the residual stream) on top of the same FIRE-equipped
    foundation, not silent HP drift. The treatment stacks
    `use_canon_conv=True` on top: one causal depthwise Conv1d
    (kernel=3) per block on the residual stream, immediately
    before the attention sublayer's pre-LN, with a single scalar
    output gate `g` (init 0 → step-0 ≡ no-conv baseline). Pre-LN
    read (no extra LN on the conv path). Strictly orthogonal to
    FIRE (additive on logits) and to CoPE/FoX/Softpick (all live
    inside the attention computation); this is an *outside-
    attention* local-mixing lever on the residual stream — the
    Griffin/Mamba local-mixing half. Default off → baseline path
    bit-identical. Cost: n_layers × (3·d_model + 1) extra params
    (~2.3K at tiny1m3m, +0.25%).

    PASS ≤ −0.01 vs the FIRE-equipped ctrl. NULL band |Δ| ≤ 0.01.
    DRIFT > +0.01. See
    `autoresearch/ideas/023-canon-conv/plan.md`.
    """
    use_fire_pe: bool = True
    use_canon_conv: bool = True


@dataclass
class Tiny1M3MUNetSigmoidOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + U-Net sigmoid skips (modded-nanogpt fix).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:44`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates
    the U-Net swap (residual-stream architectural lever) on top of
    the same FIRE-equipped foundation, not silent HP drift.

    Adds learnable U-Net skip connections bridging early layer
    outputs into mirrored late layers. The gate parameter is
    initialised to -1.5 and wrapped in sigmoid (modded-nanogpt
    PR #125 fix:
    https://github.com/KellerJordan/modded-nanogpt/pull/125), so
    `sigmoid(-1.5) ≈ 0.18` of the early activation flows in at
    step 0 — small, bounded to (0, 1), non-zero starting point
    with non-zero gradient. Categorically distinct from our
    previous broken attempt
    (`docs/youtube-architecture-ablation-log.md §5`, val +0.0003
    worse) which used `unet_gate_type="raw"` +
    `unet_gate_init=0.0` — the dead-gate bug. The mechanism never
    actually ran in that test; it was a bug-experiment, not a
    mechanism A/B.

    At tiny1m3m's 6-layer depth the U-Net mirrors are 0↔5 / 1↔4 /
    2↔3 — only 3 short pairs, so the predicted effect is "small
    but non-zero", not big-if-true. A clean null after the fix
    definitively closes U-Net skips for this model class; a win
    plausibly amplifies at 135M where depth grows. Strictly
    orthogonal to FIRE (which is an attention-side lever);
    orthogonal to all closed levers (no residual-stream
    architectural change in closed.md). transfer-risk: low —
    modded-nanogpt's +1.25% speedup is at ≥100M parameter scale,
    directly comparable to tiny1m3m's model class.

    PASS ≤ −0.005 vs the FIRE-equipped ctrl (taste's "small but
    non-zero" prediction; not −0.01 because the 3-pair U at 6L
    is a smaller bet than the deeper-stack version). NULL band
    |Δ| < 0.005. DRIFT > +0.005. See
    `autoresearch/ideas/030-unet-skip-sigmoid/plan.md`.
    """
    use_fire_pe: bool = True
    use_unet_skips: bool = True
    unet_gate_type: str = "sigmoid"
    unet_gate_init: float = -1.5


@dataclass
class Tiny1M3MGatedAttnOnFireConfig(Tiny1M3MConfig):
    """Tiny1M3M with FIRE + Gated Attention (Qiu et al. 2025).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val 6.3234
    per `closed.md:40`). The treatment stacks `use_gated_attn=True` on
    top: a per-head *scalar* input-conditional sigmoid gate on the head
    output `o_h = A_h V_h`, applied post-AV and pre-merge with the O
    projection: `o_h ← o_h · 2·σ(W_g·x+b)`. `W_g : nn.Linear(d_model, H)`
    (per-head scalar, NOT the per-head vector form — vector would be
    42% of the 0.94M model). Gate input is the sublayer input residual
    `x` (pre-LN), NOT `o_h` itself (circularity). Identity-init: W=0,
    b=0 → 2·σ(0) = 1.0 exactly at step 0, so the gated forward graph
    is bit-identical to the no-gate forward graph at step 0; the new
    params start receiving gradient from step 1. Categorically
    distinct from every closed lever and every active attention-side
    lever (020-FoX → A-prob decay, 021-V-residual → cross-layer V,
    022-softpick → softmax swap, 023-canon-conv → pre-attn conv,
    025-SSMax → logit temperature) — 024 is the *only* lever on the
    post-AV head-output value site. 009's additive position bias is
    additive on logits; the head-output gate is multiplicative on
    `o_h`; the two compose cleanly when both are on.

    PASS ≤ −0.01 vs the FIRE-equipped ctrl. NULL band |Δ| < 0.01.
    DRIFT > +0.01. See
    `autoresearch/ideas/024-gated-attention/plan.md`.
    """
    use_fire_pe: bool = True
    use_gated_attn: bool = True


@dataclass
class Tiny1M3MExclusiveSelfAttnConfig(Tiny1M3MConfig):
    """Tiny1M3M with exclusive self-attention correction.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). After standard
    attention, subtract the component of the head output that points along
    the current token's value vector. The per-head coefficient is zero-init,
    so step 0 is the baseline graph.
    """
    use_exclusive_self_attn: bool = True


@dataclass
class Tiny1M3MDropPathConfig(Tiny1M3MConfig):
    """Tiny1M3M with DropPath / Stochastic Depth (Huang et al. 2016).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    `drop_path_max=0.1` matches the original paper default and ViT-B/16
    12L. The first block is never dropped (p_1 = 1.0); only later
    blocks are candidates. Single coin flip per block per step,
    shared across the batch. The drop-path branch lives in
    `TransformerBlock.forward` and only fires when
    `self.training=True`.

    PASS ≤ ctrl − 0.005 (small/null band — 12L is at the shallow end
    of the published drop-path literature, so taste puts leverage at
    the low end). NULL band |Δ| < 0.005. DRIFT > +0.005. See
    `autoresearch/ideas/111-drop-path/idea.md`.
    """
    use_drop_path: bool = True
    drop_path_max: float = 0.1


@dataclass
class Tiny1M3MLookaheadConfig(Tiny1M3MConfig):
    """Tiny1M3M with Lookahead Optimizer Wrapper (Zhang et al. 2019).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Wraps the *list* of inner optimizers (Muon, AdamW): every k=5
    inner steps, the slow weights pull halfway toward the fast
    weights (`slow ← slow + α·(fast − slow)`, α=0.5) and the fast
    weights are reset to `slow`. Inner optimizer momentum buffers are
    cleared at the outer step so the next inner step doesn't see
    stale gradients from before the slow reset. The wrapper sits
    *outside* `optimizer.step()` in the training loop — it does not
    touch the per-step math of Muon or AdamW, only the trajectory
    shape. Identity at step 0: slow = θ_init, first inner step is
    the baseline Muon/AdamW path.

    PASS ≤ ctrl − 0.005 (taste's mid-band for a trajectory-smoothing
    wrapper at 12L depth; paper effect is small at this scale but the
    `k=5` cycle length matches the inner step count of the warmup
    phase so a null is informative). NULL band |Δ| < 0.005. DRIFT
    > +0.005. See `autoresearch/ideas/112-lookahead-opt/idea.md`.
    """
    use_lookahead: bool = True
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5


@dataclass
class Tiny1M3MSAMConfig(Tiny1M3MConfig):
    """Tiny1M3M with SAM: Sharpness-Aware Minimization
    (Foret et al. 2020, arXiv:2010.01412, ICLR 2021).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Wraps the AdamW path (1-D / embedding / norm) with `AdamSAM`:
    on every step, do an adversarial ascent to `w + ε̂` (ε̂ = ρ ·
    ∇L(w) / ‖∇L(w)‖), re-run a forward+backward at the perturbed
    point, then apply AdamW to the perturbed-point gradient. The
    Muon 2-D path is unchanged (Muon steps on the w-grad, not the
    perturbed grad — the SAM perturbation is scoped to the AdamW
    bucket only). `sam_rho=0.05` is the paper default for Adam-
    SAM (Foret et al. 2020 §5; Kwon et al. 2023 §3.2).

    At step 0 the perturbation is non-zero (O(ρ) along the
    gradient direction), so the first-step gradient differs from
    AdamW by ~5% in magnitude along the steepest-ascent axis.
    The lever's inherent first-step cost of one extra backward
    pass is the price of the flatness regularization. The 2x
    backward cost halves throughput at large scale; at 0.94M
    (~92 steps) it is ~free.

    PASS ≤ ctrl − 0.005 (small/null band — taste puts leverage at
    the low end for a 92-step trajectory where the loss surface
    may already be smooth enough that the ρ-ball contains no
    useful adversarial information). NULL band |Δ| < 0.005.
    DRIFT > +0.005 (the doubled backward cost hurts more than
    the flatness helps at this scale). See
    `autoresearch/ideas/119-sam/idea.md`.
    """
    use_sam: bool = True
    sam_rho: float = 0.05


@dataclass
class Tiny1M3MLookSAMConfig(Tiny1M3MConfig):
    """Tiny1M3M with LookSAM: Periodic SAM (Du et al. 2022, ICLR 2023,
    arXiv:2205.13539). Compute-efficient variant of SAM (119).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Wraps the AdamW path (1-D / embedding / norm) with `LookSAM`:
    every K-th step, do an adversarial ascent to `w + ε̂`, re-run a
    forward+backward at the perturbed point, then apply AdamW to the
    perturbed-point gradient. The K-1 steps in between are plain
    AdamW on the w-grad. The Muon 2-D path is unchanged. With
    `looksam_k=5` and `looksam_rho=0.05` (paper defaults), effective
    compute is ~1.2x vs. SAM's 2x — the lever keeps the flatness
    regularization at ~80% of SAM's gain while halving the overhead.

    Identity at step 0: with K=5 the first 4 steps are plain AdamW
    (`step_count=0..3`), so the first-step gradient is bit-identical
    to AdamW. The first SAM ascent fires at `step_count=4` (the
    5th step). This is *more* bit-identical at step 0 than full SAM
    (119), which runs the ascent on the first step.

    PASS ≤ ctrl − 0.005 (mid-band — periodic SAM is a regularization
    lever at 12L depth, the periodic form is the compute-efficient
    variant; null if the flatness benefit needs every-step sharpness
    information at this scale). NULL band |Δ| < 0.005. DRIFT >
    +0.005 (the periodic SAM overhead hurts more than the flatness
    helps at 0.94M). See `autoresearch/ideas/138-looksam/idea.md`.
    """
    use_looksam: bool = True
    looksam_k: int = 5
    looksam_rho: float = 0.05


@dataclass
class Tiny1M3MRDropConfig(Tiny1M3MConfig):
    """Tiny1M3M with R-Drop: KL-Regularized Dropout (Liang et al. 2021).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). For every
    train step, run the model forward **twice** with different dropout
    masks, take the mean of the two next-token CE losses, and add a
    symmetric KL penalty `α · 0.5·(KL(p_1‖p_2)+KL(p_2‖p_1))` to
    regularize the model toward dropout-invariant logits. `α` is
    linearly warmed from 0 → 1.0 over the first 1000 steps so the
    step-0 loss is `(CE_1+CE_2)/2` — bit-identical to the single-CE
    baseline modulo the doubled forward (well within run-to-run
    variance). Doubled forward means ~2× per-step wall-clock at
    0.94M, acceptable. Eval stays single-forward plain CE.

    PASS ≤ ctrl − 0.005 (taste's mid-band for a regularization lever
    at 12L depth; paper gains are small at this scale but the
    mechanism is scale-free). NULL band |Δ| < 0.005. DRIFT > +0.005.
    See `autoresearch/ideas/115-rdrop/idea.md`.
    """
    use_rdrop: bool = True
    rdrop_alpha: float = 1.0
    rdrop_warmup_steps: int = 1000


@dataclass
class Tiny1M3MRoVGatedConfig(Tiny1M3MConfig):
    """Tiny1M3M with Gated Rotary Value Embeddings (RoV).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Applies the same rotary position embedding used on Q,K to the value
    vector V as well, mixed in via a per-block scalar gate
    `rov_gate = nn.Parameter(torch.zeros(1))` (init 0 ⇒
    `V_combined = V + 0·V_rot = V`, step-0 bit-identical to baseline).
    The base rotary buffer is reused (no new params beyond the 1
    scalar/block = 12 scalars at tiny1m3m). Cost: one extra rotary
    call + one elementwise add on V per block per forward — cheap
    (≈1% of the d_model²·T FFN cost). Stays in the standard
    SDPA path (no manual attention branch needed).

    Strictly orthogonal to every closed and active lever: not QK
    rotation (009-FIRE is the rotary baseline), not V sharing
    (021-value-residual is cross-layer), not V modulation (closed
    value-channel gates). It is the only lever on the *intra-layer V
    position* axis. When `use_nope`/`use_cope` is on, RoV is a no-op
    (the geometric lever is unavailable). NULL band |Δ| < 0.005 (the
    lever is unverified on language modeling — paper wins come from
    Hunyuan-DiT / SD3-style image generation; the bet is that the
    same V-position-blindness failure mode transfers). DRIFT > +0.005
    (the extra rotary call genuinely costs something at our tier).
    PASS ≤ −0.005. See `autoresearch/ideas/151-rov-gated/idea.md`.
    """
    use_rov: bool = True


@dataclass
class Tiny1M3MHyperConnectionsConfig(Tiny1M3MConfig):
    """Tiny1M3M with Hyper-Connections (Xie et al. 2024, arXiv:2409.19606).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Splits the residual stream into `hc_n_resid=4` parallel streams of
    width `d_l = 16` each, with per-position `(A_l, B_l, C_l) ∈ R^{4×4}`
    mixing matrices (48 scalars/position × 12 = 576 total). Identity init
    (A=B=C=I) ⇒ step-0 forward graph is bit-identical to the pre-norm
    residual path. The model has to *learn* to mix streams during
    training. Strictly orthogonal to every closed and active lever in
    the queue: it is the only mechanism that *expands* the residual
    stream itself, not a regularization or attention-side tweak.

    Sub-LN and DropPath already closed as nulls at 6L/12L, and the
    bet is at the small end of the paper's reported effect (DeepSeek-V3
    headline wins are at 100L+). The slot tests whether the
    multi-stream *capacity* lever survives the 12L regime even when
    the regularization levers don't.

    PASS ≤ ctrl − 0.005 (mid-band for an architectural expansion at
    6L/12L; mid-band because the prior closed-nulls of sub-LN and
    DropPath suggest residual-stream levers often don't fire at this
    depth, but the multi-stream *expansion* lever is qualitatively
    different — capacity, not regularization). NULL band |Δ| < 0.005.
    DRIFT > +0.005 (stream-mixing overhead hurts more than it helps).
    See `autoresearch/ideas/116-hyper-connections/idea.md`.
    """
    use_hyper_connections: bool = True
    hc_n_resid: int = 4


@dataclass
class Tiny1M3MMARSConfig(Tiny1M3MConfig):
    """Tiny1M3M with MARS Variance-Reduced AdamW (Yuan et al. 2024).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Routes the 1-D / embedding / norm / head path to `MARSAdamW` (a
    thin subclass of `torch.optim.AdamW` that adds a lag-based
    variance-reduction correction `g̃_t = g_t + mix_coef *
    (m_{t-lag} − m_{t-2*lag})` to the *gradient* input). The 2-D
    Muon path is unchanged. At step 0 the ring buffer is empty ⇒
    correction undefined ⇒ g̃_t = g_t ⇒ bit-identical to plain
    AdamW for the first `2*lag=20` steps. The paper recommends
    `lag=10`, `mix_coef=0.5` (Yuan et al. 2024 §3.2 / Table 1);
    the LR is the same as the parent AdamW (no re-tuning
    required).

    PASS ≤ ctrl − 0.005 (small/null band — taste puts leverage at
    the low end for a 92-step trajectory where AdamW's per-
    parameter v noise is already partially damped by the long
    warmup-decay schedule). NULL band |Δ| < 0.005. DRIFT > +0.005
    (the correction adds variance, not removes it, at this scale).
    See `autoresearch/ideas/114-mars/idea.md`.
    """
    use_mars: bool = True
    mars_lag: int = 10
    mars_mix_coef: float = 0.5
    mars_lr_scale: float = 1.0


@dataclass
class Tiny1M3MProdigyConfig(Tiny1M3MConfig):
    """Tiny1M3M with Prodigy: Parameter-Free AdamW (Mishchenko & Defazio 2023).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Routes the 1-D / embedding / norm / head path to `Prodigy`. The
    2-D Muon path is unchanged (Prodigy is an AdamW replacement, not
    a Muon replacement). Prodigy maintains a group-level step-size
    estimate `D_t` updated each step from a continuous Adam-style
    gradient similarity `s_t = ⟨sign(g_t/√v_t), sign(g_{t-k}/√v_{t-k})⟩`
    via `D ← D · exp(β3·s_t)` — eliminating D-Adaptation's noisy
    binary ramp-up. The first `prodigy_warmup_steps=10` steps are
    unit-LR AdamW (D = d0 = 1.0) and `D_0` is then reset to
    `‖w_0 − w_k‖ / k` — a *displacement-based* warm-start that puts
    D in the right ballpark from step 11 onward.

    `prodigy_d0=1.0` is the warm-start scalar (NOT the production
    LR — the production LR is `D_t`, which Prodigy discovers). The
    user's `lr` parameter to Prodigy is more like a "unit
    conversion" between AdamW's update and the trajectory; we
    default it to `1.0` (paper default). The first 10 steps move
    the model by `‖w_0 − w_k‖` units of AdamW-LR=1, so `D_0 ≈
    ‖Δ‖/10` which is the natural step size for that trajectory.

    At tiny1m3m's 92-step budget the 10-step warmup is ~11% of the
    run — non-trivial. The bet is that Prodigy's smoother ramp-up
    wins at this scale because it eliminates the early LR
    misallocation. A null would say "the LR ramp-up is a small
    fraction of total loss at 0.94M"; a win would say "every step
    at tiny1m3m matters and Prodigy uses the first 10 steps
    better than hand-tuned `adamw_lr=0.006`".

    PASS ≤ ctrl − 0.005 (small/null band — taste's mid-band for an
    LR-removal lever at 12L depth; the lever is about ramp-up
    *quality*, not raw capacity). NULL band |Δ| < 0.005. DRIFT
    > +0.005. See `autoresearch/ideas/121-prodigy/idea.md`.
    """
    use_prodigy: bool = True
    prodigy_d0: float = 0.01      # 100× below paper default; the previous
                                  # d0=1.0 caused the 2026-06-13 GPU blowup
                                  # (12.01 → 10348 at step 25). With d0=0.01
                                  # the first 10 warmup steps are in the same
                                  # ballpark as a hand-tuned AdamW lr=0.006.
    prodigy_warmup_steps: int = 10
    prodigy_beta3: float = 0.01
    prodigy_d_max: float = 1.0    # paper §3.1 default; bounds the
                                  # discovery loop. Required for stability
                                  # at tiny1m3m (prevents unbounded D
                                  # growth on a sign-agreement run).
    prodigy_min_d: float = 1e-6   # lower clamp on D.
    prodigy_update_clip: float = 1.0  # per-param max-norm on the
                                       # in-place step.


@dataclass
class Tiny1M3MEMAConfig(Tiny1M3MConfig):
    """Tiny1M3M with Polyak-Ruppert Weight EMA (Polyak 1990; RoBERTa,
    MAE, MoCo v3, modded-nanogpt speedrun SWA).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Maintains a shadow copy of the live parameters updated each step
    as `θ_ema ← μ·θ_ema + (1−μ)·θ` with `μ` ramping linearly from 0
    to 0.999 over the first 100 steps. At step 0 `μ=0` ⇒ `θ_ema = θ_live`,
    so the val score at step 0 is bit-identical to the baseline.
    `ema_eval_only=True` keeps `θ_live` as the saved/resumed model and
    only swaps in the EMA for the val pass — training and
    checkpointing stay on the live trajectory.

    The mechanism is one line per step (`ema_params[n].mul_(μ).add_(p,
    alpha=1-μ)`) and a swap-in/swap-out around `evaluate_model(...)`.
    Cost: a clone of every trainable parameter (~0.94M floats at
    tiny1m3m), no per-step math beyond one in-place EMA update.

    PASS ≤ ctrl − 0.005 (small/null band — taste puts leverage at the
    low end for a 92-step trajectory where the per-step
    signal-to-noise is already high). NULL band |Δ| < 0.005.
    DRIFT > +0.005 (the EMA averaging would slow convergence and
    raise val loss if the lever is hurting). See
    `autoresearch/ideas/110-weight-ema/idea.md`.
    """
    use_ema_eval: bool = True
    ema_decay: float = 0.999
    ema_warmup_steps: int = 100
    ema_eval_only: bool = True


@dataclass
class Tiny1M3MMegaConfig(Tiny1M3MConfig):
    """Tiny1M3M with Mega EMA on V (Ma et al. 2022, arXiv:2209.10655).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    The V stream is concatenated with `V_ema = W_V @ u` where
    `u_t = β·u_{t-1} + (1-β)·x_t` is a per-channel EMA over the input
    residual stream. β ∈ [0, 1] is parametrized via sigmoid of a
    learnable per-channel scalar (`mega_beta_raw`, init 0 ⇒ β=0.5 at
    step 0). The EMA is computed once per layer via a depthwise causal
    conv1d over the T axis with kernel `(1-β)·β^k`. The concat doubles
    the V stream from `[B, T, kv_size]` to `[B, T, 2·kv_size]` and the
    head reshape treats this as 2·n_kv_heads heads (asserted == n_heads
    at tiny1m3m: 2·2 = 4). Same W_V slice is reused for V_ema (zero
    new projection params). Cost: 1 d_model-dim parameter/layer for
    the β buffer (12 × 64 = 768 floats at tiny1m3m, ~0.1%). NOT byte-
    identical to baseline at step 0 — the EMA at β=0.5 is half-
    smoothed (not the current token), and the concat doubles V.

    PASS ≤ ctrl − 0.005 (mid-band for an attention-side V smoothing
    lever at 12L depth; the EMA's effect is on the per-token value
    stream, which the standard softmax attention already covers via
    the K·V dot product, so the lever is plausibly a wash at 0.94M).
    NULL band |Δ| < 0.005. DRIFT > +0.005 (the doubled V stream adds
    compute for no gain at this scale). See
    `autoresearch/ideas/134-mega-ema/idea.md`.
    """
    use_mega: bool = True
    mega_beta: float = 0.9
    mega_use_input: bool = True


@dataclass
class Tiny1M3MGaLoreConfig(Tiny1M3MConfig):
    """Tiny1M3M with GaLore: Gradient Low-Rank Projection (Zhao et al. 2024).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the Muon 2-D non-embed, non-norm slot with GaLore: each
    2-D weight matrix's gradient is projected into a rank-4 subspace
    via orthonormal P, Q; AdamW runs in the 4×4 projected space; the
    update is projected back. P, Q refresh from the SVD of a running
    gradient EMA every 200 steps (paper defaults). 1-D / embedding /
    norm stay on plain AdamW. The forward graph is unchanged, so
    step-0 val_loss is bit-identical to baseline.

    `galore_rank=4` is the paper's sweet spot at the 1B tier; at
    0.94M the per-parameter second-moment noise is already small
    so the lever's effect is at the small end. A null would say the
    heavy-tail gradient structure is not load-bearing at this scale;
    a win would compound with the closed-WIN stack (Muon orth +
    AdamW 1-D). A regression would say the rank-r constraint is
    actively hurting — a known failure mode if the gradient's top-r
    components are *not* where AdamW is needed (e.g. if
    `use_galore` is stacked with muon or with an already-orthogonal
    optimizer on the same slot).

    PASS ≤ ctrl − 0.005 (taste's mid-band for an orthogonal
    optimizer-side lever at 12L depth; paper effect is small at this
    scale but the memory-side lever is a no-op at 0.94M and the
    *quality* side is what we're testing). NULL band |Δ| < 0.005.
    DRIFT > +0.005. See `autoresearch/ideas/113-galore/idea.md`.
    """
    use_galore: bool = True
    galore_rank: int = 4
    galore_proj_every: int = 200
    galore_lr: float = 0.006
    galore_beta1: float = 0.9
    galore_beta2: float = 0.999
    galore_eps: float = 1e-8


@dataclass
class Tiny1M3MSoftMoEConfig(Tiny1M3MConfig):
    """Tiny1M3M with Soft MoE FFN replacement (Puigcerver et al. 2024).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the standard dense FFN with `SoftMoEFFN` (E=4 parallel
    narrower FFNs + softmax dispatch/combine). Each expert has
    width `d_ff / 4 = 64` (was 256), so total FFN params stay at the
    budget (E × 2·d_model·d_ff_e = 2·d_model·d_ff, matching the
    baseline). The dispatch/combine are derived from a small per-token
    linear projection (`W_d, W_c` of shape `[m·E, d_model]`); zero-init
    ⇒ uniform softmaxes at step 0 ⇒ all experts see the same weighted
    average of input tokens and the layer collapses to ~a single FFN
    applied to `mean(X)`.

    Fully differentiable: no top-k, no balancing loss, no straight-
    through. The only MoE lever filed is 108-simbal-router (which is a
    *router regularizer* on hard routing — a different mechanism
    requiring hard routing infrastructure first). 117 is the cleanest
    possible MoE extension: tests the **MoE capacity hypothesis** at
    0.94M with the fewest confounders.

    PASS ≤ ctrl − 0.005 (taste's small/null band — at 0.94M the per-
    expert FFN is `d_ff / E = 64`, below the standard `d_ff ≥ 4·d_model`
    rule of thumb, so the lever is at the small end of the paper's
    reported effect). NULL band |Δ| < 0.005. DRIFT > +0.005. See
    `autoresearch/ideas/117-soft-moe/idea.md`.
    """
    use_soft_moe: bool = True
    soft_moe_n_experts: int = 4
    soft_moe_n_slots: int = 4


@dataclass
class Tiny1M3MMixtureOfDepthsConfig(Tiny1M3MConfig):
    """Tiny1M3M with Mixture-of-Depths (Raposo et al. 2024).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Each transformer block gets a small `MoDRouter` (2-layer MLP,
    `d_model=64 → 64 → 1`) that scores every token, and the top-k
    (`k = mod_capacity · T = 1024` at default) tokens get the block's
    residual update. Skipped tokens are passed through unchanged. The
    kept tokens' residual update is rescaled by `c = k/T = 0.5` so the
    expected per-token contribution matches the dense baseline. W_1, W_2
    are zero-init ⇒ σ(0) = 0.5 uniform scores ⇒ step-0 top-k is an
    arbitrary subset (no signal yet). NOT byte-identical to baseline at
    step 0 (expected residual magnitude per token = 0.5·E[block(x)],
    not E[block(x)]) but the deviation is bounded and explicit; with
    `use_mod=False` (default) the `MoDRouter` is never built and the
    baseline path is bit-identical.

    PASS ≤ ctrl − 0.005 (taste's small/null band — at 0.94M / 12L the
    paper's effect is at the low end: the gains are largest at 24L+
    where the router has more skip decisions to learn from). NULL band
    |Δ| < 0.005. DRIFT > +0.005 (the routing overhead hurts more than
    it helps at shallow depth). See
    `autoresearch/ideas/118-mixture-of-depths/idea.md`.
    """
    use_mod: bool = True
    mod_capacity: float = 0.5
    mod_router_hidden: int = 64


@dataclass
class Tiny1M3MKDAChannelGateConfig(Tiny1M3MConfig):
    """Tiny1M3M with KDA per-channel diagonal V-gate (bounded).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Replaces the
    closed unbounded per-(head, channel) V-gate
    (`use_value_channel_gate`, `1+g` form) with a *bounded* `2·σ(g)` per
    channel — KDA's `Γ = diag(γ_1, …, γ_d)` diagonal-decay idea,
    ported to the softmax-attention V stream. `g ∈ R^{n_heads × d_k}`
    zero-init ⇒ `2·σ(0) = 1.0` exactly at step 0 ⇒ step-0 is the
    baseline graph. Applied to V before the AV product
    (orthogonal site to every active attention-side lever; same
    site as the closed `use_value_channel_gate`, but the bounded
    parametrization is the difference). Cost: n_heads × d_k = 64
    scalars per layer × 12 layers = 768 extra params (~0.08%).

    PASS ≤ −0.01 vs the tiny1m3m ctrl. NULL band |Δ| < 0.01.
    DRIFT > +0.01. See
    `autoresearch/ideas/109-kda-channel-gate/idea.md`.
    """
    use_kda_channel_gate: bool = True


@dataclass
class Tiny1M3MDAdaptConfig(Tiny1M3MConfig):
    """Tiny1M3M with D-Adaptation: Automatic LR Discovery (Defazio 2023).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the AdamW 1-D / embedding / norm / head path with
    `DAdaptAdamW` (a thin subclass of `torch.optim.AdamW` that
    maintains a per-group scalar `D` and derives the effective LR as
    `lr_t = D / ‖g_t‖`). The 1st/2nd moments of AdamW are retained
    intact — only the outer LR scaling is swapped. Muon 2-D path is
    unchanged (D-Adapt is ortho to Muon, lives only on the AdamW
    bucket). At step 0 `D = 1e-6` warm-start ⇒ `lr_0 ≈ 1e-6 / ‖g_0‖`
    (essentially zero); after ~10–20 steps `D` reaches a typical
    AdamW-equivalent value. The first ~10 steps see a *ramp-up* in
    effective LR — this is the lever's signature, not a bug (the
    design sketch explicitly accepts this). The lever is NOT bit-
    identical to AdamW at step 0 (different LR), but the first-step
    displacement is O(ε) and within run-to-run noise; with `D` frozen
    at its initial value the lever collapses to AdamW.

    `dadapt_d0_lr=1.0` and `dadapt_min_lr=0.0` are the paper's
    defaults (Defazio 2023 §3). The lever's signature at 0.94M is
    informative either way: a win would mean the config's
    `adamw_lr=0.006` is suboptimal and the discovery loop finds a
    better one; a null would mean the LR ramp-up cost (~10% of the
    92-step trajectory) outweighs any LR-discovery win.

    PASS ≤ ctrl − 0.005 (the win-bar from the design sketch — the
    gain comes from removing a small LR misconfiguration, not from a
    new direction of improvement). NULL band |Δ| < 0.005. DRIFT
    > +0.005 (the ramp-up cost hurts more than the discovery helps
    at this scale). See
    `autoresearch/ideas/120-dadaptation/idea.md`.
    """
    use_dadapt: bool = True
    dadapt_d0_lr: float = 1.0
    dadapt_min_lr: float = 0.0
    dadapt_d_max: float = 1.0    # paper §3.1 default; required for
                                 # stability at tiny1m3m (prevents
                                 # unbounded D-growth → val blowup)


@dataclass
class Tiny1M3MAdaShiftConfig(Tiny1M3MConfig):
    """Tiny1M3M with AdaShift: Decorrelated Adam via Delayed Gradients.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the AdamW 1-D / embedding / norm / head path with
    `AdaShift` (Zhou et al. 2019, arXiv:1810.00143). The 2-D Muon
    path is unchanged (AdaShift is an AdamW replacement, like
    114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam).
    The update uses a *delayed* gradient `g_{t-n}²` for the 2nd
    moment, decorrelating `v_t` from `m_t`:
        m_t = β1·m_{t-1} + (1-β1)·g_t
        v_t = β2·v_{t-1} + (1-β2)·g_{t-n}²
        update = m̂_t / (√v̂_t + ε)

    Per-parameter state keeps a queue of past `n` gradients (fp32
    clones, length bounded by n). The paper's warm-start
    `v_0 = g_0²` makes `v_1 = β2·g_0²` — NOT bit-identical to
    AdamW's first step (`v_1 = (1-β2)·g_0²`) but same magnitude
    order (O(β2) different). The first-step displacement is the
    lever, not a bug. With `n = 0` AdaShift collapses to AdamW;
    `adashift_n = 3` is the paper's recommended delay.

    PASS ≤ ctrl − 0.005 (mid-band for an AdamW-replacement lever at
    12L depth; the bet is on `m_t`/`v_t` decorrelation that
    AdamW's coupled second moment doesn't provide). NULL band
    |Δ| < 0.005. DRIFT > +0.005 (the queue overhead + first-step
    displacement cost outweigh any decorrelation benefit at this
    scale). See `autoresearch/ideas/126-adashift/idea.md`.
    """
    use_adashift: bool = True
    adashift_lr: float = 0.006
    adashift_beta1: float = 0.9
    adashift_beta2: float = 0.999
    adashift_eps: float = 1e-8
    adashift_n: int = 3


@dataclass
class Tiny1M3MVResidualOnFireConfig(Tiny1M3MConfig):
    """Tiny1M3M with FIRE + Value Residual Learning (cross-layer V shortcut).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:44`; ctrl spread 6.3875–6.4050 per
    `closed.md:41-44`). The treatment stacks `use_value_residual=True`
    on top: stash the projected V at layer 0 (post-W_V, post-GQA
    repeat_interleave, post-transpose, shape `[B, n_heads, T, d_k]`);
    in every later layer l > 0, blend
    `V_l ← (1 - λ_l)·V_l + λ_l·V_1` BEFORE `attn_weights @ V`, with
    `λ_l = nn.Parameter(torch.zeros(()))` per-block on MHA (identity-
    init at step 0 ⇒ baseline-bit-identical at flag-on, step 0).
    `.detach()` on the V_1 stash ⇒ each layer's W_V trains on its own
    attention path; the cross-layer shortcut only learns the blend
    weight, not the layer-0 projection. Strictly orthogonal to FIRE
    (which is *additive* on logits): FIRE chooses *which* key wins;
    021 changes *which* value-stream the winners read from. The bet
    is that tiny1m3m's narrow heads (d_k=32 at H=8) suffer attention
    concentration, and a direct shortcut to the first-layer value
    representation gives later blocks a cleaner value signal.

    Categorically distinct from the closed V/Q/K/O *embedding* axis
    (input-side projection scaling, an added embedding to the value
    *source*; 021 is a cross-layer residual on the value *stream*)
    and from every active attention-side lever (020-FoX = post-softmax
    A·D, 022-softpick = softmax swap, 024-gated-attn = post-AV o_h
    gate, 025-SSMax = logit temperature) — 021 is the only lever on
    the projected V stream.

    PASS ≤ −0.005 vs the FIRE-equipped ctrl (low-to-moderate bar; the
    bet is at the small end of the paper's reported effect — the
    taste r1 reviewer asked for exactly this band). NULL band
    `|Δ| < 0.01` (sub-noise; the lever does not fire on top of FIRE
    at this scale). DRIFT > +0.01 (cross-layer mix hurts attention
    concentration rather than helping). See
    `autoresearch/ideas/021-value-residual/plan.md`.
    """
    use_fire_pe: bool = True
    use_value_residual: bool = True


@dataclass
class Tiny1M3MQKNormOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + QK-Norm (LayerNorm on Q,K head-dim).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:44`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates the
    QK-Norm swap (per-head LayerNorm on Q,K along `d_head`) on top
    of the same FIRE-equipped foundation, not silent HP drift. The
    treatment stacks `use_qk_layernorm=True` on top: bounds the
    per-head logit `Q·K/√d_head` to `|·| ≤ √d_head`. Categorically
    distinct from FIRE — FIRE is *additive* (bias added to logits
    post-dot-product); QK-Norm is *multiplicative-normalizing*
    (LayerNorm bounds the dot-product magnitude that the bias gets
    added to). The two operate at different points and on different
    mathematical axes.

    The 013-CoPE DRIFT (+0.143 vs FIRE-alone, `closed.md`) is the
    relevant prior, but 013 failed by stacking *two additive position
    bias levers* — QK-Norm does not compound additively with FIRE's
    bias; it bounds the magnitude of the Q·K product. This is the
    qualitative difference that makes 026 a different bet from 013.

    Expected: additive (~−0.078 vs FIRE-alone, computed as 009's
    −0.064 + 016's −0.014). Superadditive (~−0.09+) would mean the
    per-head logit bounding makes FIRE's learned position bias more
    consistent across heads. A null or regression would mean the
    013-CoPE precedent generalises — attention-domain headroom is
    exhausted by FIRE at this scale.

    PASS ≤ −0.01 vs the FIRE-equipped ctrl. NULL band |Δ| ≤ 0.01.
    DRIFT > +0.01. See
    `autoresearch/ideas/026-fire-x-qknorm/plan.md`.
    """
    use_fire_pe: bool = True
    use_qk_layernorm: bool = True


@dataclass
class Tiny1M3MDeepThinConfig(Tiny1M3MConfig):
    """Tiny1M3M deep-and-thin: depth/width swap at fixed ~0.94M budget.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306
    per `LEADERBOARD.md` row 14). The treatment reallocates the 0.94M
    budget across more, thinner layers: n_layers 12→20 (1.67×),
    d_model 64→48 (0.75×), d_ff 256→192 (= 4·d_model preserved),
    n_heads/n_kv_heads 4/2 → 3/3 (MHA-tied — see confound below).
    Per-head `d_head = 16` preserved (was 64/4, now 48/3); `emb_rank=8`,
    `ffn_variant="squared_relu"`, `vocab_size=49152` all inherited
    unchanged from `Tiny1M3MConfig`. Param budget arithmetic:
    per-block attn 9.2k + FFN 18.4k + norms 0.24k ≈ 27.9k; ×20 = 558k
    + embedding factorisation 393.6k ≈ 951k (+1.3% vs baseline 939k,
    inside the ±5% ceiling). MobileLLM (Ma et al., ICML 2024,
    arXiv:2402.14905) reports +2.7% / +4.3% on zero-shot benchmarks
    at 125M / 350M from this exact depth/width swap; the open question
    is whether the lever still fires at 0.94M (133× smaller than the
    paper's smallest ablation).

    Known confound (see `idea.md:50-55`). Baseline is GQA 2:1
    (n_heads=4, n_kv_heads=2). The depth/width swap also collapses
    kv-sharing → MHA (n_heads=n_kv_heads=3). Tied-QK / full-MHA is a
    known WIN signature at tiny1m3m (`LEADERBOARD.md` row 0 = vq-gain
    + rope250k + swa384 + tiedqk, val 6.3041) — the trt Δ partly
    reflects the kv-sharing collapse, not pure depth/width. Picked B1
    over B1' (MQA n_kv_heads=1) and B2 (d_model=32, d_ff off the
    4·d_model rule) because the `d_ff = 4·d_model` convention is more
    load-bearing for "pure depth/width swap" than the GQA ratio.
    Runner reports the confound alongside the raw val-loss Δ.

    PASS ≤ ctrl − 0.01 (clears the cited ±0.01 box-noise floor).
    NULL band |Δ| ≤ 0.01 (inclusive — sub-noise = inconclusive,
    no multi-seed rescue). DRIFT > ctrl + 0.01. ctrl_val baseline
    6.4306 (`LEADERBOARD.md` row 14) — interpreted against the
    in-session ctrl run to avoid cross-session drift. Seed 42 only.
    See `autoresearch/ideas/028-deep-thin-config/plan.md`.
    """
    d_model: int = 48
    n_heads: int = 3
    n_kv_heads: int = 3
    n_layers: int = 20
    d_ff: int = 192


@dataclass
class Screen10M20MSwiGLUConfig(Screen10M20MConfig):
    """Screen10M20M with SwiGLU feed-forward blocks."""
    ffn_variant: str = "swiglu"
    d_ff: int = 384  # Parameter-matched to squared-ReLU d_ff=576.


@dataclass
class Screen10M20MOutputAdapterConfig(Screen10M20MConfig):
    """Screen10M20M with a rank-32 additive output adapter.

    Tests whether the rank-48 tied factorized softmax is too narrow after the
    embedding/depth reallocation. Adds ~1.58M parameters while staying under
    the 10M class.
    """
    output_adapter_rank: int = 32


@dataclass
class Screen10M20MSmearGateConfig(Screen10M20MConfig):
    """Screen10M20M with SmearGate previous-token embedding blend."""
    use_smear_gate: bool = True


@dataclass
class Screen10M20MUNetSkipConfig(Screen10M20MConfig):
    """Screen10M20M with zero-init U-Net skip bridges across depth."""
    use_unet_skips: bool = True


@dataclass
class Screen10M20MAttnOutputGateConfig(Screen10M20MConfig):
    """Screen10M20M with per-head attention-output gates."""
    use_attn_output_gate: bool = True


@dataclass
class Screen10M20MLayerScaleConfig(Screen10M20MConfig):
    """Screen10M20M with per-channel attention/MLP LayerScale gates."""
    use_layerscale: bool = True


@dataclass
class Screen10M20MValueEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token value embeddings injected into attention V."""
    use_value_embed: bool = True


@dataclass
class Screen10M20MQueryEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token query embeddings injected into attention Q."""
    use_query_embed: bool = True


@dataclass
class Screen10M20MKeyEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token key embeddings injected into attention K.

    The natural mirror of #29/#30. K goes through RoPE downstream, so the
    projection's term is positionally rotated — a different operating point
    from V (no RoPE) and Q (also RoPE'd). The cheapest probe of whether
    the token-identity-into-attention lever has more headroom in the K
    direction.
    """
    use_key_embed: bool = True


@dataclass
class Screen10M20MVQEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token value + query embeddings injected into attention.

    #32 — combination probe. V is the end-of-training winner (#29, 4.7728),
    Q is the fast-warmup winner (#30, 4.8159). Tests whether the lever is
    additive (Q's warmup advantage + V's end-game edge) or whether V's
    V-specific position is the unique story. Cost = V-embed + Q-embed
    projections = ~166k extra params (~2% over baseline).
    """
    use_value_embed: bool = True
    use_query_embed: bool = True


@dataclass
class Screen10M20MOutputEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token embeddings injected into the attention OUTPUT.

    #33 — fundamentally different lever. Where #29-#32 inject the raw
    token embedding into the attention INPUTS (V/Q/K, inside the score
    computation), this one injects it into the attention OUTPUT (after
    the O projection, straight into the residual stream). The token
    identity bypasses attention entirely. Tests "is V-embed winning
    because V is a unique position, or because any token-signal-to-
    residual helps?" Most likely outcome: underperforms V-embed (since
    the signal bypasses attention) but a clean probe of an architectural
    question we haven't asked yet. Cost = 24 × d_model 144 × emb_rank
    48 = 165,888 extra params (~2.1%).
    """
    use_output_embed: bool = True


@dataclass
class Screen10M20MVQKEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token value + query + key embeddings injected into attention.

    #34 — full combo probe. V-embed alone is the natural-end winner
    (4.7728), Q-embed adds 0.03 to V (V+Q = 4.7428). K-embed is
    essentially tied with Q at the natural end (4.8228 vs 4.8159,
    inside noise). Tests "is K redundant with Q in the V+Q combo?"
    If V+Q+K ≈ V+Q, K adds nothing. If V+Q+K < V+Q, K is hurting.
    If V+Q+K > V+Q, K is contributing beyond Q.

    Cost = 24 × (q_size 144 + 2 × kv_size 48) × emb_rank 48
        = 24 × 240 × 48 = 276,480 extra params (~3.6% over baseline).
    """
    use_value_embed: bool = True
    use_query_embed: bool = True
    use_key_embed: bool = True


@dataclass
class Screen10M20MVOEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token value (inside attention) + output (post-O) embeddings.

    #35 — across-boundary combo. V-embed wins inside attention (4.7728),
    O-embed is the worst of the family (4.8350). Tests whether the
    inside-attention and post-O positions are additive — i.e. whether
    adding the token signal to the residual stream helps when V is
    already injecting it into attention. If V+O < V, the residual
    signal interferes with V's inside-attention signal. If V+O > V,
    the residual signal adds value.

    Cost = 24 × (kv_size 48 + d_model 144) × emb_rank 48
        = 24 × 192 × 48 = 221,184 extra params (~2.9% over baseline).
    """
    use_value_embed: bool = True
    use_output_embed: bool = True


@dataclass
class Screen10M20MVOKEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with value + output + key embeddings.

    #36 — K's role across contexts. K is anti-additive in V+Q (4.8250 vs
    V+Q's 4.7428). Tests whether K is universally bad or just bad in
    the V+Q context. With O replacing Q (since V+O = best so far at
    4.7188), K has a different gradient environment. If V+O+K > V+O,
    K helps when paired with O. If V+O+K < V+O, K is universally
    bad in any embed combo.

    Cost = 24 × (kv_size 48 + kv_size 48 + d_model 144) × emb_rank 48
        = 24 × 240 × 48 = 276,480 extra params (~3.6% over baseline).
    """
    use_value_embed: bool = True
    use_output_embed: bool = True
    use_key_embed: bool = True


@dataclass
class Screen10M20MQGainConfig(Screen10M20MConfig):
    """Screen10M20M with per-head learnable Q-gain (post-RoPE).

    #37 — first non-embed lever. Each attention head has a learnable
    scalar that multiplies its Q vector after norm+RoPE. Zero-init so
    step 0 is exact baseline. Equivalent to per-head temperature on
    the attention scores. Known modded-nanogpt speedrun trick
    (q_gain). Cost: 24 × 6 = 144 extra params (negligible).

    If q_gain helps, the model benefits from per-head attention
    temperature — a way for different heads to specialize. If
    q_gain is in noise, the heads don't need to rescale their
    attention patterns.
    """
    use_q_gain: bool = True


@dataclass
class Screen10M20MKGainConfig(Screen10M20MConfig):
    """Screen10M20M with per-head learnable K-gain (post-RoPE).

    #42 — symmetric to q_gain but on K. Tests whether scaling K
    helps as much as scaling Q. If k_gain is similar to q_gain,
    the lever is "per-head temperature" in general. If only q_gain
    helps, it's specifically the Q side.
    """
    use_k_gain: bool = True


@dataclass
class Screen10M20MVQKGainConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + per-head Q-gain + per-head K-gain.

    #43 — V+q+k_gain. Tests whether q_gain and k_gain are
    additive. If V+q+k_gain < V+q_gain, k_gain is hurting or
    redundant. If V+q+k_gain > V+q_gain, k_gain adds value.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_k_gain: bool = True


@dataclass
class Screen10M20MQKGainConfig(Screen10M20MConfig):
    """Screen10M20M with per-head Q-gain + per-head K-gain (no embed).

    #44 — pure double-gain (no embed). Tests if the Q-gain and
    K-gain levers together beat either alone.
    """
    use_q_gain: bool = True
    use_k_gain: bool = True


@dataclass
class Screen10M20MDeepVEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with 2-layer non-linear V-embed (GELU bottleneck).

    #45 — deep V-embed probe. Tests whether the linear V-embed (#29,
    4.7728) has saturated at a single projection, or whether a
    non-linear "bottleneck" V-embed has more capacity. The
    architecture:

        V += GELU(ve @ W1) @ W2
        W1: [emb_rank=48, hidden=96]  zero-init
        W2: [hidden=96, kv_size=48]   zero-init

    Both zero-init so step 0 = exact baseline. GELU has a dead-zone
    at 0 so the first gradient step only flows through W2 (similar
    to standard deep ResNets).

    Cost = 24 × (48 × 96 + 96 × 48) = 24 × 9,216 = 221,184 extra
    params (+2.9% over baseline, +1.4% over V-embed).

    If deep V-embed > V-embed (linear), the V-embed win has more
    capacity to unlock. If deep V-embed ≈ V-embed, the linear
    projection is already sufficient. If deep V-embed < V-embed,
    the non-linearity hurts (likely due to overfitting or gradient
    issues with the dead-zone at init).
    """
    use_deep_value_embed: bool = True
    deep_value_embed_hidden: int = 96


@dataclass
class Screen10M20MDeepVQGainConfig(Screen10M20MConfig):
    """Screen10M20M with deep V-embed + per-head Q-gain.

    #46 — combines #45 (deep V-embed) with #37 (q_gain). Tests
    whether the q_gain lever is also additive with the deeper
    V-embed architecture. If V_deep+q_gain > V+q_gain (4.6815),
    deep V-embed is the new capacity ceiling.
    """
    use_deep_value_embed: bool = True
    deep_value_embed_hidden: int = 96
    use_q_gain: bool = True


@dataclass
class Screen10M20MFFNEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token embeddings injected into FFN input.

    #47 — new position probe. The FFN-embed adds a learned projection
    of the factorized token embedding to the FFN input (post-attention,
    pre-FFN, after norm2). Different path from V-embed (in attention)
    and O-embed (post-O residual). The FFN now has direct access to
    token identity without going through attention.

    Cost = 24 × (d_model 144 × emb_rank 48) = 165,888 extra params
    (~2.1% over baseline).

    Tests:
    - If FFN-embed ≈ V-embed (4.7728), the lever is "token identity
      into residual content" regardless of position.
    - If FFN-embed > V-embed, the FFN is a more useful position than
      attention's V (because it's a more direct path).
    - If FFN-embed < V-embed, the position matters — V-embed's win is
      specifically about attention content, not residual content.
    """
    use_ffn_embed: bool = True


@dataclass
class Screen10M20MVQGFFNEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + FFN-embed.

    #48 — combines #29 (V-embed), #37 (q_gain), and #47 (FFN-embed).
    Tests whether the FFN-embed lever is also additive with V+q_gain.
    If V+q_gain+ffn_embed < V+q_gain (4.6815), FFN-embed conflicts
    with V+q_gain. If V+q_gain+ffn_embed > V+q_gain, FFN-embed adds
    a new dimension to the win.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_ffn_embed: bool = True


@dataclass
class Screen10M20MVQGainQKPostNormConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + QK-norm-post-RoPE.

    #49 — applies the modded-nanogpt QK-norm-post-RoPE variant on top
    of V+q_gain. Different mathematical operating point: the post-RoPE
    norm constrains post-RoPE Q,K magnitudes per head. Flag-only, no
    extra params. Tests whether the normalization story (where the
    norm is applied) breaks the V+q_gain plateau.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_qk_norm_post_rope: bool = True


@dataclass
class Screen10M20MVQGainSwiGLUConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + SwiGLU FFN.

    #50 — combines V+q_gain with SwiGLU FFN (instead of squared_relu).
    SwiGLU is a different FFN activation. Tests whether the FFN
    activation is part of the V+q_gain plateau. Uses d_ff=384 so the
    3-matrix SwiGLU FFN is parameter-matched to squared-ReLU d_ff=576:
    3 * d_model * 384 == 2 * d_model * 576.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    ffn_variant: str = "swiglu"
    d_ff: int = 384


@dataclass
class Screen10M20MVQGainSlidingWindowConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + sliding-window attention.

    #51 — first attention-pattern axis in the ladder. V+q_gain (the
    plateau) plus a local causal window of 512 tokens. Flag-only, no
    extra params. Tests whether the attention *pattern* (not just the
    inputs) has headroom at this scale. Window 512 = quarter of
    seq_len 2048, a clean first probe. If this beats V+q_gain
    (4.6815), the attention matrix itself was a hidden lever; if it
    ties, long-range is a wash; if it loses, long-range is load-
    bearing and SWA is closed.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512


@dataclass
class Screen10M20MSlidingWindowConfig(Screen10M20MConfig):
    """Screen10M20M with sliding-window attention ONLY (no embeds, no gains).

    #52 — clean ablation. Same window (512) as #51, but the V-embed and
    q_gain levers are off. Tests whether sliding-window attention is a
    standalone lever (in which case this should land near V+q+SWA's
    4.6700) or just a small add-on to V+q_gain (in which case this
    should land near control 4.7984, or even worse if long-range is
    load-bearing). The most informative single screen for deciding
    whether the architecture change is "use SWA" or "use V+q_gain".
    """
    use_sliding_window: bool = True
    sliding_window_size: int = 512


@dataclass
class Screen10M20MNoPEConfig(Screen10M20MConfig):
    """Screen10M20M with no positional encoding (NoPE).

    #53 — fresh axis: positional encoding. Skips the rotary call
    entirely while keeping the Q/K RMSNorm. Flag-only, no extra
    params. Tests whether RoPE is load-bearing at this scale. If
    NoPE ≈ control (4.7984), position is mostly conveyed by the
    causal mask + token identity injection (and our V-embed lever
    is partially substituting for RoPE). If NoPE << control, RoPE
    is critical and there's no slack there. If NoPE < control,
    position is hurting — surprising but worth measuring.
    """
    use_nope: bool = True


@dataclass
class Screen10M20MVQGainNoPEConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + NoPE.

    #54 — tests whether NoPE is additive with the V+q_gain plateau
    (3-seed mean 4.6815). If V+q+NoPE < 4.6815, NoPE is a real lever
    on the best baseline. If V+q+NoPE > 4.6815, RoPE is load-bearing
    for V+q and NoPE is closed. If V+q+NoPE ≈ 4.6815, position is a
    wash when paired with V+q_gain.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_nope: bool = True


@dataclass
class Screen10M20MLayerTied2Config(Screen10M20MConfig):
    """Screen10M20M with layer tying (ALBERT-style, group_size=2).

    #55 — fresh axis: weight sharing across depth. 24 layers, every
    group of 2 consecutive blocks shares weights, so 12 unique
    TransformerBlock modules are used twice each. Drops unique
    depth params by ~50% (qkvo+FFN per block). Tests whether depth
    uniqueness or depth *re-use* matters more at this scale. If
    layer_tied ≈ control (4.7984), unique depth is critical. If
    layer_tied < control, weight sharing acts as cheap
    regularization.
    """
    tie_layer_groups: int = 2


@dataclass
class Screen10M20MVQGainLayerTied2Config(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + layer tying (group_size=2).

    #56 — combines V+q_gain (the plateau, 3-seed mean 4.6815) with
    layer tying (group_size=2, 12 unique blocks). Tests whether the
    V+q lever still works when each block is used twice — i.e.
    whether V-embed projections and q_gain scalars can survive the
    weight-sharing constraint. If V+q+tied < V+q, layer tying is
    additive with V+q; otherwise it conflicts.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    tie_layer_groups: int = 2


@dataclass
class Screen10M20MGQA1Config(Screen10M20MConfig):
    """Screen10M20M with aggressive GQA (n_kv_heads=1).

    #57 — fresh axis: GQA ratio. The base config has n_kv_heads=2
    (each KV head shared across 3 Q heads). This drops it to 1 (6:1
    GQA — every Q head reads from the same K/V). Fewer KV params,
    more aggressive sharing. Tests whether the GQA ratio is a real
    architecture lever at this scale. If GQA1 < base, the model
    benefits from more attention-head sharing; if GQA1 > base, KV
    diversity is load-bearing.
    """
    n_kv_heads: int = 1


@dataclass
class Screen10M20MMHAConfig(Screen10M20MConfig):
    """Screen10M20M with full multi-head attention (n_kv_heads=n_heads).

    #58 — the other end of the GQA axis: no sharing at all.
    n_kv_heads=6 means each Q head has its own K and V projection.
    More KV params, no information sharing between heads. Tests
    whether the current 3:1 GQA is a wash (in which case MHA ≈ GQA)
    or whether more KV capacity helps. This is the "no GQA" point.
    """
    n_kv_heads: int = 6


@dataclass
class Screen10M20MVQGainMHAConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + full MHA (n_kv_heads=6).

    #59 — combines V+q_gain (the plateau, 3-seed mean 4.6815) with
    full MHA. Tests whether the GQA ratio is additive with V+q_gain
    on the best baseline. If V+q+MHA < V+q, full MHA is a real lever
    on top of the plateau; otherwise GQA is sufficient.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    n_kv_heads: int = 6


@dataclass
class Screen10M20MGELUConfig(Screen10M20MConfig):
    """Screen10M20M with GELU FFN activation (no gating, no squaring).

    #60 — fresh axis: MLP activation. The base config uses
    squared_relu (Primer-style); SwiGLU was tried and washed (#50).
    This is plain GELU, the most common transformer activation.
    Single up-projection, no gating, parameter-matched to
    squared_relu d_ff=576. Tests whether the activation is itself
    a real architecture lever — a cleaner test than SwiGLU, which
    differs in BOTH activation AND number of projections.
    """
    ffn_variant: str = "gelu"


@dataclass
class Screen10M20MVQGainGELUConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + GELU FFN.

    #61 — combines V+q_gain (the plateau, 3-seed mean 4.6815) with
    GELU FFN. Tests whether the activation swap is additive with
    V+q. If V+q+GELU < V+q, GELU is a real lever; otherwise the
    activation is a wash when V+q is in play.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    ffn_variant: str = "gelu"


@dataclass
class Screen10M20MVQGainSWAGELUConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + sliding-window + GELU FFN.

    #62 — combines the current best screen20m levers (V+q+SWA at
    4.6700 single-seed) with the only untried MLP activation
    (GELU). Tests whether GELU is additive with the V+q+SWA
    plateau. If V+q+SWA+GELU < 4.6700, GELU is the new best add-on.
    If V+q+SWA+GELU > V+q+SWA, GELU conflicts.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    ffn_variant: str = "gelu"


@dataclass
class Screen10M20MHighRoPEConfig(Screen10M20MConfig):
    """Screen10M20M with Llama-style RoPE base (500000).

    #63 — fresh axis: positional decay. The default base=10000 gives
    short wavelength (positional information blurs fast). Llama's
    500000 keeps positional information sharper over longer
    distances. Tests whether our seq_len=2048 is hitting the edge
    of the default RoPE's useful range.
    """
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPEConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + Llama-style RoPE base.

    #64 — combines the new best screen20m levers (V+q+SWA at
    4.6676 2-seed mean) with the high RoPE base. Tests whether
    Llama-style positional decay is additive with V+q+SWA.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPEGELUConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + High RoPE + GELU FFN.

    #65 — combines the new best screen20m levers (V+q+SWA+
    HighRoPE at 4.6364) with GELU FFN. Tests whether GELU is
    additive with the V+q+SWA+HighRoPE plateau.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    ffn_variant: str = "gelu"


@dataclass
class Screen10M20MVQGainSWAHighRoPETied2Config(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + High RoPE + layer tying (group=2).

    #66 — V+q+LayerTied2 was anti-additive (4.7419 vs V+q 4.6797),
    but on the new best baseline (V+q+SWA+HighRoPE 4.6364) the
    question is whether SWA+RoPE-base have changed the
    regularization story enough that tying now adds value.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    tie_layer_groups: int = 2


@dataclass
class Screen10M20MVQGainSWAHighRoPEMHAConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + High RoPE + full MHA (n_kv_heads=6).

    #67 — MHA alone was a wash on control (4.7981 vs 4.7984).
    On the new best baseline the question is whether the GQA
    ratio becomes a lever when paired with the other wins.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    n_kv_heads: int = 6


@dataclass
class Screen10M20MVQGainHighRoPESWA256Config(Screen10M20MConfig):
    """Screen10M20M with V+q + High RoPE + SWA (window=256).

    #68 — window size sweep on the V+q+HighRoPE plateau. The current
    best baseline uses window=512. Tests whether a smaller window
    (256 = 1/8 of seq_len) is better. Smaller window = more
    aggressive locality. Mask density: sum(min(256,i+1))/2048^2 =
    0.13.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 256
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainHighRoPESWA1024Config(Screen10M20MConfig):
    """Screen10M20M with V+q + High RoPE + SWA (window=1024).

    #69 — window size sweep (larger). Window=1024 = half of
    seq_len. Tests whether the default window=512 is sub-optimal.
    Mask density: sum(min(1024,i+1))/2048^2 = 0.378.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 1024
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainHighRoPENoSWAConfig(Screen10M20MConfig):
    """Screen10M20M with V+q + High RoPE + NO SWA.

    #70 — V+q+HighRoPE without SWA. Tests whether SWA is still
    load-bearing on the new RoPE-base=500000 baseline. If this
    lands at ~4.6364, SWA is redundant on top of HighRoPE. If
    it's much worse, SWA is still load-bearing.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = False
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPELogitSoftcapConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + logit softcap (Gemma-style).

    #71 — logit softcap=15.0 on the new best baseline. Real
    architecture change: logit = softcap * tanh(logit/softcap).
    Tests whether the cap is a real lever. Gemma uses 30.0; we
    test 15.0 because our model is smaller.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    logit_softcap: float = 15.0


@dataclass
class Screen10M20MVQGainSWATiedQKConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + Tied QK (PaLM-style).

    #72 — Tied QK: Q and K share the same projection matrix.
    Real arch change. PaLM uses this as the default attention
    design. Tests whether tying QK weights is a real lever.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    use_tied_qk: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPETiedQKConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + Tied QK (PaLM-style).

    #72b — same as #72 but on the V+q+SWA+HighRoPE best baseline.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_tied_qk: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPEMLAConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + MLA (DeepSeek-V2-style).

    #73 — Multi-head Latent Attention: K,V are computed via a
    low-rank latent (d_c=mla_latent_dim, default d_model//4=36).
    Real arch change. DeepSeek-V2 uses this. Tests whether
    the latent bottleneck is a real lever on our small model.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_mla: bool = True
    mla_latent_dim: int = 36


@dataclass
class Screen10M20MVQGainSWAHighRoPEDilatedConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + Dilated Attention.

    #74 — dilation=2. Same window_size=512 by token count, but
    positions are spread (every other position in the window
    range). Tests whether strided patterns beat contiguous
    locality at this scale. Effective range: 2*512=1024 tokens.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    attention_dilation: int = 2


@dataclass
class Screen10M20MVQGainSWAHighRoPEPostNormConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + post-norm.

    #75 — fundamental arch change: norm goes AFTER the residual
    addition (original Transformer) instead of before (modern
    pre-norm). Tests whether post-norm is a real lever at
    our depth=24.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_post_norm: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPEGQA1Config(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + GQA=1 (max GQA).

    #76 — n_kv_heads=1 means every Q head reads from the same
    K,V. On the best baseline — does max GQA add or hurt?
    GQA1 standalone (#76) was bad; the question is whether
    the best baseline changes that.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    n_kv_heads: int = 1


@dataclass
class Screen10M20MVQGainSWAHighRoPENoEmbScaleConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + no embedding scale.

    #77 — the standard code multiplies the token embedding by
    sqrt(d_model). Set to 1.0 (no scaling). Tests whether the
    standard scaling is a hidden knob at this scale.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    embedding_scale: float = 1.0


@dataclass
class Screen10M20MVQGainSWAHighRoPESWAFullConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + SWA=seq_len (full).

    #78 — window=2048 = seq_len. Effectively full causal attention
    but with the SWA code path. The cleanest "is SWA helping at
    all" test on the best baseline.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 2048
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPELayerNormConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + LayerNorm (vs RMSNorm).

    #79 — RMSNorm is the default. LayerNorm is the older
    alternative with learned bias. Tests whether the choice
    of norm is a real lever on the best baseline.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_layernorm: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPELinearAttnConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + linear attention.

    #80 — Performer-style linear attention. Replaces the
    softmax(QK^T / sqrt(d_k))V with phi(Q) (phi(K)^T V) where
    phi(x) = elu(x) + 1. Different attention math (O(n) instead
    of O(n^2) in the full case, but windowed in our case).
    Tests whether linear attention unlocks a new operating point
    on the best baseline.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_linear_attn: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPEQKPostNormConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + QK norm after RoPE.

    #81 — modded-nanogpt style Q/K normalization position, but on
    the current best baseline. V+q alone tied with this knob; the
    question is whether SWA+HighRoPE changes the operating point.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_qk_norm_post_rope: bool = True


@dataclass
class Screen10M20MVQGainHighRoPESWA384Config(Screen10M20MConfig):
    """Screen10M20M with V+q + HighRoPE + SWA(window=384).

    #82 — finer locality sweep between the losing 256 window and
    the current best 512 window.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 384
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainHighRoPESWA768Config(Screen10M20MConfig):
    """Screen10M20M with V+q + HighRoPE + SWA(window=768).

    #83 — finer locality sweep between the current best 512 window
    and the losing 1024 window.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 768
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPE250KConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA(window=512) + RoPE base 250k.

    #84 — finer RoPE-base sweep. Default 10k lost, 500k won; this
    checks whether the optimum sits below 500k.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 250000


@dataclass
class Screen10M20MVQGainSWAHighRoPE1MConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA(window=512) + RoPE base 1M.

    #85 — finer RoPE-base sweep above the current 500k winner.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 1000000


@dataclass
class Screen10M20MVOQGainConfig(Screen10M20MConfig):
    """Screen10M20M with V+O + per-head Q-gain. Best embed (V+O 4.7188)
    + non-embed lever (q_gain). Tests whether q_gain is additive
    with V+O.
    """
    use_value_embed: bool = True
    use_output_embed: bool = True
    use_q_gain: bool = True


@dataclass
class Screen10M20MVQGainConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + per-head Q-gain (no O-embed).

    #39 — partial-ablation probe. Best arch (V+O+q_gain = 4.6789 mean
    across 3 seeds) drops the O-embed. Tests whether O-embed is the
    necessary addition or whether q_gain alone is enough to push
    V-embed down. If V+q_gain ≈ V+O+q_gain, O is unnecessary. If
    V+q_gain >> V but V+q_gain << V+O+q_gain, O is the load-bearing
    piece.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True


@dataclass
class Screen10M20MVQQGainConfig(Screen10M20MConfig):
    """Screen10M20M with V+Q + per-head Q-gain (no O-embed).

    #40 — V+Q+q_gain is an alternative to V+O+q_gain. Tests
    whether V+O is the unique best embed combo, or whether V+Q
    with q_gain also helps. If V+Q+q_gain ≈ V+O+q_gain, O and Q
    are interchangeable (just a different place to put the embed).
    If V+Q+q_gain > V+O+q_gain, Q is the better add-on.
    """
    use_value_embed: bool = True
    use_query_embed: bool = True
    use_q_gain: bool = True


@dataclass
class Screen10M20MQGainConfig(Screen10M20MConfig):
    """Screen10M20M with per-head Q-gain (no embeds).

    #41 — pure non-embed lever. Tests whether q_gain is the
    load-bearing piece (if so, this should land near V+O+q_gain).
    If q_gain alone is in noise, q_gain is only additive WITH
    the embeds.
    """
    use_q_gain: bool = True


# ============================================================================
# FULL ladder — 20x tokens (compute-optimal / Chinchilla). Transfer-valid: this
# is where a mechanism's real optimum is locked. Ladder 10M→25M→50M→135M lets
# you fit optimum-vs-size and extrapolate to the 135M release target. Same
# architecture at every size (RoPE + GQA + RMSNorm + squared-ReLU + Muon);
# scaling is hyperparameters + engineering, not an architecture change.
# Param counts use tied embeddings (vocab 49,152).
# ============================================================================


@dataclass
class Full10M200MConfig(LLMConfig):
    """Ladder — ~7.7M params · 200M tokens (20x) · ~48,800 steps. The 10m record target.

    The 10M architecture: low-rank embedding (emb_rank=48) + depth (24 layers),
    same shape as Screen10M20MConfig but trained to the 20x regime — the cheapest
    transfer-valid point, runnable locally. First rung of the release ladder.
    """
    d_model: int = 144
    n_heads: int = 6
    n_layers: int = 24
    d_ff: int = 576
    n_kv_heads: int = 2
    emb_rank: int = 48
    max_seq_len: int = 2048
    batch_size: int = 2
    train_tokens: int = 200_000_000
    compile_model: bool = False
    warmup_ratio: float = 0.02
    schedule_type: str = "warmup_decay_to_zero"
    eval_milestones: Optional[Tuple[int, ...]] = tuple(range(0, 48800, 2000))


@dataclass
class Full10M200MOutputAdapterConfig(Full10M200MConfig):
    """Full10M200M with a rank-32 additive output adapter."""
    output_adapter_rank: int = 32


@dataclass
class Full10M200MSmearGateConfig(Full10M200MConfig):
    """Full10M200M with SmearGate previous-token embedding blend."""
    use_smear_gate: bool = True


@dataclass
class Full10M200MUNetSkipConfig(Full10M200MConfig):
    """Full10M200M with zero-init U-Net skip bridges across depth."""
    use_unet_skips: bool = True


@dataclass
class Full10M200MAttnOutputGateConfig(Full10M200MConfig):
    """Full10M200M with per-head attention-output gates."""
    use_attn_output_gate: bool = True


@dataclass
class Full10M200MLayerScaleConfig(Full10M200MConfig):
    """Full10M200M with per-channel attention/MLP LayerScale gates."""
    use_layerscale: bool = True


@dataclass
class Full10M200MValueEmbedConfig(Full10M200MConfig):
    """Full10M200M with token value embeddings injected into attention V."""
    use_value_embed: bool = True


@dataclass
class Full10M200MQueryEmbedConfig(Full10M200MConfig):
    """Full10M200M with token query embeddings injected into attention Q."""
    use_query_embed: bool = True


@dataclass
class Full10M200MKeyEmbedConfig(Full10M200MConfig):
    """Full10M200M with token key embeddings injected into attention K."""
    use_key_embed: bool = True


@dataclass
class Full135M2700MConfig(LLMConfig):
    """Release target — ~134.5M params · 2.7B tokens (20x). SmolLM2-135M class.

    The model we race to release: benchmark head-to-head vs SmolLM2-135M.
    """

    d_model: int = 576
    n_heads: int = 9          # head_dim 64
    n_layers: int = 30
    d_ff: int = 2304          # 4x d_model
    n_kv_heads: int = 3       # 3:1 GQA
    max_seq_len: int = 2048
    train_tokens: int = 2_700_000_000  # ~20x params (Chinchilla-optimal)


# ============================================================================
# Query-tweaks plan — 29 Screen10M20M<Name>Config recipes (Batches 1-6).
# See docs/research-plans/query-tweaks/plan.md and manifest.md.
# ============================================================================

# ---- Batch 1: high-signal levers (Q1-Q4) ----

@dataclass
class Screen10M20MAlibiBiasConfig(Screen10M20MConfig):
    """Q1 — ALiBi-style per-head distance bias. scores += -m_h·(i-j)."""
    use_alibi_bias: bool = True

@dataclass
class Screen10M20MQTempTokenConfig(Screen10M20MConfig):
    """Q2 — Token-conditioned per-head Q temperature. Q *= (1 + tanh(x·w_h))."""
    use_q_temp_token: bool = True

@dataclass
class Screen10M20MCosineAttnConfig(Screen10M20MConfig):
    """Q3 — Cosine attention. L2-normalize Q,K; per-head learnable τ."""
    use_cosine_attn: bool = True

@dataclass
class Screen10M20MQKBilinearConfig(Screen10M20MConfig):
    """Q4 — Per-channel relevance. score = Q^T diag(d_h) K (d_h init 1)."""
    use_qk_bilinear: bool = True

# ---- Batch 2: flagship + positional (Q5-Q7) ----

@dataclass
class Screen10M20MTalkingHeadsQConfig(Screen10M20MConfig):
    """Q5 — Talking-heads on Q. learned n_h×n_h M on attention logits pre-softmax."""
    use_talking_heads_q: bool = True

@dataclass
class Screen10M20MPerHeadRopeBaseConfig(Screen10M20MConfig):
    """Q6 — Per-head learnable RoPE base. θ_h init = global base."""
    use_per_head_rope_base: bool = True

@dataclass
class Screen10M20MPartialRotaryConfig(Screen10M20MConfig):
    """Q7 — Partial rotary. Rotate only 50% of Q/K dims."""
    partial_rotary_p: float = 0.5

# ---- Batch 3: exotic (Q8-Q10) ----

@dataclass
class Screen10M20MQExpansionConfig(Screen10M20MConfig):
    """Q8 — Multi-query expansion. Q += W·x (zero-init W; step-0 baseline)."""
    use_q_expansion: bool = True

@dataclass
class Screen10M20MDecoupledContentPosConfig(Screen10M20MConfig):
    """Q9 — Decoupled content/position attention (DeBERTa-style)."""
    use_decoupled_content_pos: bool = True

@dataclass
class Screen10M20MAntisymQKConfig(Screen10M20MConfig):
    """Q10 — Antisymmetric Q·K coupling. +Q^T S K with skew S (init 0)."""
    use_antisym_qk: bool = True

# ---- Batch 4: query-norm zoo (Q11-Q16) ----

@dataclass
class Screen10M20MNormPNormConfig(Screen10M20MConfig):
    """Q11 — Q-side pnorm p=1.5 (Lp norm, outlier-robust)."""
    q_norm_type: str = "pnorm1.5"

@dataclass
class Screen10M20MNormClipConfig(Screen10M20MConfig):
    """Q12 — Q-side Winsorized RMSNorm (clip k=3)."""
    q_norm_type: str = "clipnorm3"

@dataclass
class Screen10M20MNormChannelScaleConfig(Screen10M20MConfig):
    """Q13 — Q-side ChannelScale (learnable pre-scale)."""
    q_norm_type: str = "channelscale"

@dataclass
class Screen10M20MNormManhattanConfig(Screen10M20MConfig):
    """Q14 — Q-side Manhattan (L1 MAD) norm."""
    q_norm_type: str = "manhattan"

@dataclass
class Screen10M20MNormCenterConfig(Screen10M20MConfig):
    """Q15 — Q-side Center norm (mean-only, no variance)."""
    q_norm_type: str = "center"

@dataclass
class Screen10M20MNormNoneConfig(Screen10M20MConfig):
    """Q16 — Q-side norm disabled. K still normed."""
    q_norm_type: str = "none"

# ---- Batch 5: learnable-param zoo (Q17-Q23) ----

@dataclass
class Screen10M20MQPerHeadBiasConfig(Screen10M20MConfig):
    """Q17 — Per-head bias. Q += b_h (per-head×channel) post-RoPE."""
    use_q_per_head_bias: bool = True

@dataclass
class Screen10M20MQPerChannelGainConfig(Screen10M20MConfig):
    """Q18 — Per-channel gain. Q *= g_d post-RoPE."""
    use_q_per_channel_gain: bool = True

@dataclass
class Screen10M20MQHDGainConfig(Screen10M20MConfig):
    """Q19 — Head×channel gain. Q *= g_hd post-RoPE."""
    use_q_hd_gain: bool = True

@dataclass
class Screen10M20MQNormGateConfig(Screen10M20MConfig):
    """Q20 — Norm-gate. per-head scalar σ(a_h·‖x‖+b_h) on Q."""
    use_q_norm_gate: bool = True

@dataclass
class Screen10M20MQLowRankRefineConfig(Screen10M20MConfig):
    """Q21 — Low-rank refine. Q += (W1·x)@W2 (zero-init)."""
    use_q_lowrank_refine: bool = True

@dataclass
class Screen10M20MQLayerScaleConfig(Screen10M20MConfig):
    """Q22 — LayerScale on Q. Q *= (1 + ls_d) per-channel post-RoPE."""
    use_q_layerscale: bool = True

@dataclass
class Screen10M20MQSoftplusGainConfig(Screen10M20MConfig):
    """Q23 — Softplus gain. Q *= softplus(g_h) per-head — always ≥ 0."""
    use_q_softplus_gain: bool = True

# ---- Batch 6: architecture/mixing (Q24-Q29) ----

@dataclass
class Screen10M20MQHeadMixConfig(Screen10M20MConfig):
    """Q24 — Head-mix. Q ← Q + Q @ M (M−I init 0) pre-attention."""
    use_q_head_mix: bool = True

@dataclass
class Screen10M20MQTimeConvConfig(Screen10M20MConfig):
    """Q25 — Time-conv. 1D conv k=3 over position axis, zero-init."""
    use_q_time_conv: bool = True

@dataclass
class Screen10M20MQEMASmoothConfig(Screen10M20MConfig):
    """Q26 — EMA-smooth over position. Q ← α·Q + (1−α)·Q_{t-1}."""
    use_q_ema_smooth: bool = True

@dataclass
class Screen10M20MQFeatureMapConfig(Screen10M20MConfig):
    """Q27 — Feature-map attention. NOT identity-init — needs own control."""
    use_q_feature_map: bool = True

@dataclass
class Screen10M20MQPerTokenRopeConfig(Screen10M20MConfig):
    """Q28 — Per-token RoPE. Each token's θ via small MLP on x."""
    use_q_per_token_rope: bool = True

@dataclass
class Screen10M20MQNoiseRegConfig(Screen10M20MConfig):
    """Q29 — Noise reg. Q += N(0, σ²) training only (learnable σ)."""
    use_q_noise_reg: bool = True


# =====================================================================
# Cautious-Muon recipes — appended here (not above) because they
# reference classes defined later in the file. Single-line addition to
# the Muon optimizer step (Liang et al. 2024, arXiv 2411.16085):
# zero out the orthogonalized update where its sign disagrees with the
# current gradient. See optimizers/muon.py for the implementation.
# =====================================================================


@dataclass
class Screen10M20MCautiousMuonConfig(Screen10M20MConfig):
    """Screen10M20M with cautious-muon sign-mask + small LR bump.

    A/B vs the screen20m control (4.8487) — should land ≤ 4.8387 for a pass.
    """
    use_cautious_muon: bool = True
    muon_lr: float = 0.025


@dataclass
class Screen10M20MVQGainSWAHighRoPECautiousMuonConfig(Screen10M20MVQGainSWAHighRoPEConfig):
    """V+q+SWA+HighRoPE best baseline + cautious-muon sign-mask.

    A/B vs the current screen20m best (4.6364) — tests whether cautious-Muon
    is additive on top of the V+q+SWA+HighRoPE plateau. Multi-seed confirm
    if single-seed wins.
    """
    use_cautious_muon: bool = True
    muon_lr: float = 0.025


@dataclass
class Tiny1M3MAdamPConfig(Tiny1M3MConfig):
    """Tiny1M3M with AdamP: Adam + projection-based update (He et al. 2020).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the AdamW 1-D / embedding / norm / head path with `AdamP`,
    which projects the Adam update `Δ = m̂/√v̂` onto the orthogonal
    complement of `w` (removing the component of Δ along w) so the
    update rotates direction without changing magnitude. The L2 reg
    is applied as the paper's `λ · ‖w‖ · ŵ` (pure magnitude
    shrinkage, no rotation). The 2-D Muon path is unchanged.

    Identity at step 0: for symmetric inits the projection removes
    an `O(1/√d)` component of Δ_0, so the first AdamP step ≈ the
    first AdamW step modulo that small correction. With
    `adamp_lambda=0.0` the projection is fully inert and `AdamP`
    collapses to plain AdamW. With `use_adamp=False` (default)
    plain `torch.optim.AdamW` is used — baseline bit-identical.

    PASS ≤ ctrl − 0.01 (Δ ≤ −0.01). NULL band |Δ| < 0.01.
    DRIFT > +0.01. See `autoresearch/ideas/137-adamp/idea.md`.
    """
    use_adamp: bool = True


@dataclass
class Tiny1M3MLayerScaleConfig(Tiny1M3MConfig):
    """Tiny1M3M with LayerScale (Touvron et al. 2021, arXiv:2103.17239).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Per-block per-channel learnable diagonal scale on the residual
    branch, applied in the *direct* form `x = x + gamma * sub_block(x)`
    (NOT the reparam `(1+γ)` form used by the closed-#21 `use_layerscale`
    flag). Init `gamma = 1e-4 * ones(d_model)` (paper default) → at
    step 0 the residual contribution is `1e-4 × sub_block(x)`, four
    orders of magnitude smaller than the residual stream magnitude,
    so the val loss at step 0 is within fp32 noise of baseline. The
    per-channel selectivity is qualitatively different from the scalar
    ReZero (130) and the whole-residual Sub-LN (017) — all four prior
    depth-conditional levers null at 12L, but LayerScale's per-channel
    diagonal is a structurally different mechanism that has not been
    tested in this pipeline. Cost: 2 × d_model = 128 extra params at
    tiny1m3m (negligible).

    Transfer-risk: med — paper's headline wins are at depth ≥ 50, with
    smaller gains at 12L. NULL band |Δ| < 0.01. DRIFT > +0.01.
    PASS ≤ −0.01. See `autoresearch/ideas/142-layerscale/idea.md`.
    """
    use_layer_scale: bool = True
    layer_scale_init: float = 1e-4


@dataclass
class Tiny1M3MMoSConfig(Tiny1M3MConfig):
    """Tiny1M3M with Mixture of Softmaxes output head
    (Yang, Chen, et al. 2017, arXiv:1711.03953, "Breaking the Softmax
    Bottleneck: A High-Rank RNN Language Model").

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the single output softmax with `n_mos_components=2` parallel
    vocab-sized heads mixed by per-token `π = softmax(W_π · h)`. The
    mixture distribution is
        `P(v) = Σ_k π_k · softmax(W_k · h)[v]`,
    computed in log space as
        `log P(v) = logsumexp_k (log π_k + log_softmax(W_k · h))`.

    The structural lever is the *rank* of the output distribution: a
    single `softmax(W·h)` has log-prob matrix rank ≤ d_model = 64, but a
    K-mixture has effective rank ≤ K·d_model = 128 — twice as
    expressive for high-rank next-token targets. The existing output
    head (`d_model × vocab`) is replaced by K fresh vocab-sized heads
    (no tying with token_embedding) plus a small `d_model → K` mix
    projection.

    `n_mos_components=2` (paper default is K=4) — recoded from K=4 in
    round 2 because K=4 with the chunked-B*T forward still OOM'd on
    the RTX 3060 12GB. The OOM culprit was the downstream
    `F.cross_entropy(logits.view(-1, V), …)` materializing a (4096,
    49152) tensor in fp32 (≈ 3 GiB). Combined with K=4's 3 fresh
    vocab-sized heads (~37.8M extra params + AdamW state ≈ 450 MB)
    the trainer process exceeded 12 GB. Halving K cuts the MoS
    optimizer state to ~150 MB, restoring headroom. The effective
    rank lever still doubles (`K·d_model = 128 vs 64`), so the A/B is
    still informative — just at a smaller K.

    Identity at step 0 (strict, fp32-exact):
    - `W_π.weight = 0`, `W_π.bias = [+1e4, -1e4]` ⇒
      `softmax(W_π·h) = [1, 0]` exactly (the `exp(-2e4)` term
      underflows to 0 in fp32). The `logsumexp` then reduces to
      `log_softmax(W_0 · h)`, which equals the baseline tied head's
      output logit. So the step-0 val loss is bit-identical to the
      baseline. The K-1=1 fresh head `W_1` is init'd to `N(0, 0.02²)`
      like the rest of the model; only head 0 contributes at step 0.

    Cost: K × vocab × d_model = 2 × 49152 × 64 = 6,291,456 extra
    params (the lever's headline param injection), plus K × d_model
    = 128 for the mix projection. At tiny1m3m the baseline has 0.94M
    params; MoS treatment has ~7.2M — still a sizeable param confound
    that the paper's win at billion-param scale does NOT control
    for. The transfer note should flag this when judging the A/B.

    Transfer-risk: med. Paper trains RNN-based 1B+ LMs with K=4
    softmaxes; independent Transformer replications at 100M+ report
    ~0.1-0.3 perplexity gains. tiny1m3m is well below the validated
    range and the K× param cost is non-trivial. NULL band |Δ| < 0.01.
    DRIFT > +0.01. PASS ≤ −0.01. See `autoresearch/ideas/144-mos/idea.md`.
    """
    use_mos: bool = True
    n_mos_components: int = 2


@dataclass
class Tiny1M3MExpertChoiceConfig(Tiny1M3MConfig):
    """Tiny1M3M with Expert-Choice MoE FFN replacement
    (Zhou, Lei, et al. 2022, arXiv:2202.09368,
    "Mixture-of-Experts with Expert Choice Routing").

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the standard dense FFN with `ExpertChoiceMoE` — E parallel
    full-width FFNs (default E=4) where each expert picks its own
    top-k tokens (k = ceil(N/E)) instead of each token picking its
    top-k experts (token-choice MoE). Load balance is by construction:
    every expert processes exactly k tokens, so NO auxiliary load-
    balancing loss is required. Router `nn.Linear(d_model, n_experts)`
    is zero-init ⇒ at step 0 all expert-token scores are 0 ⇒ every
    expert processes the same set of k tokens with uniform softmax
    weights ⇒ output ≈ uniform mean of E identically-init'd FFNs
    (NOT byte-identical to a single FFN at step 0 — documented
    caveat mirroring 117-soft-moe). Each expert is full-width so the
    FFN-param cost multiplies by `n_moe_experts` (default 4×).

    With `use_expert_choice_moe=False` (default) the `ExpertChoiceMoE`
    module is never built and the baseline FFN path is bit-identical.
    See `models/expert_choice_moe.py` and
    `autoresearch/ideas/145-expert-choice/idea.md`.

    PASS ≤ ctrl − 0.01. NULL band |Δ| < 0.01. DRIFT > +0.01.
    """
    use_expert_choice_moe: bool = True
    n_moe_experts: int = 4


@dataclass
class Tiny1M3MTTTLinearConfig(Tiny1M3MConfig):
    """Tiny1M3M with TTT-Linear FFN replacement (Sun, Yang, et al.
    2024, arXiv:2407.04620, §3.2).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces the FFN's up-projection with `TTTLinear` — a per-input
    closed-form fast-weight linear that updates its own weight from
    the input on the fly via a single Newton-style gradient step on
    the auto-encoding loss `||W·x − x||²`. The down-projection stays
    a standard `nn.Linear` so the FFN output side is unchanged.

    Per-input fast weights act as a *capacity multiplier*: the static
    FFN must encode "what to do with token t" in fixed weights, but
    TTTLinear gives the model a per-input weight update at the cost
    of one extra matmul per step. At tiny1m3m (0.94M params, 92
    update steps) this is a "free" sample-efficiency boost when the
    model is undertrained.

    Identity at step 0: `ttt_lr_init=0.0` (default) zero-inits the
    per-layer TTT learning rate so `lr=0` at step 0 ⇒ the
    `TTTLinear` short-circuits to `F.linear(x, weight, b)` with the
    same `kaiming_uniform_` weight as `nn.Linear` ⇒ the FFN is bit-
    identical to a vanilla `SquaredReLUFeedForward` at step 0. With
    `use_ttt_ffn=False` (default) the `TTTFeedForward` module is
    never built and the baseline FFN path is bit-identical. See
    `models/ttt_linear.py` and
    `autoresearch/ideas/149-ttt-linear/idea.md`.

    PASS ≤ ctrl − 0.01. NULL band |Δ| < 0.01. DRIFT > +0.01.
    """
    use_ttt_ffn: bool = True
    ttt_lr_init: float = 0.0


@dataclass
class Tiny1M3MXLayerFeedbackConfig(Tiny1M3MConfig):
    """Tiny1M3M with Cross-Layer Feedback Attention
    (Holtzman et al. 2020, arXiv:2002.09402; lean "previous K=2
    layers" variant).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Each block reads from a cache of the previous K=2 blocks' pre-FFN
    residual states via a small `XLayerCrossAttn` head (1 head,
    head_dim=16, Q/K dim=16, V=full d_model), and adds the result as
    a *gated* residual branch. Per-block scalar `xlayer_gate` is init
    0 ⇒ the contribution is exactly 0 at step 0 ⇒ the forward is
    bit-identical to the no-feedback baseline.

    The cache is plumbed by the model loop (`MinimalLLM` keeps an
    `xlayer_mem` list, appends the block's pre-FFN x after each
    forward, truncates to K). The cross-attn reads from this list
    (Q from current pre-FFN x, K/V from the previous K pre-FFN
    states concatenated along the time axis).

    Param overhead: per-block `2·d_model·16 + 2·d_model² ≈ 8.2K` at
    d_model=64. Across 12 blocks: ~98K params, ≈10.5% of the 0.94M
    budget. The cross-attn head is intentionally small (1 head × 16
    dim) to keep the per-block cost tractable; bigger K (4 or 8) is
    an obvious sweep axis.

    Distinct from value-residual (021, WIN at tiny1m3m, -0.0723):
    value-residual blends layer-0's V into every later layer's V
    (a value-only path, no attention, no selection). Cross-Layer
    Feedback is a *cross-attention* over a window of K layers'
    hidden states (a selection mechanism, not a linear mixing).
    Composability: both are *additive* residual branches on the
    V stream / residual stream respectively, so they could in
    principle stack.

    Closest null: 116-hyper-connections (mHC). mHC is *linear mixing*
    of adjacent-layer outputs (no attention, no selection). Cross-
    Layer Feedback is *attention-weighted* selection from a K=2
    window. The 116 null at 0.94M raised the bar for the
    cross-layer-info-flow axis; this is the attention variant.

    PASS ≤ ctrl − 0.01. NULL band |Δ| < 0.01. DRIFT > +0.01. See
    `autoresearch/ideas/150-xlayer-feedback/idea.md`.
    """
    use_xlayer_feedback: bool = True
    xlayer_k: int = 2


@dataclass
class Tiny1M3MRebasedAttnConfig(Tiny1M3MConfig):
    """Tiny1M3M with Rebased Attention (Shi et al. 2024, arXiv:2407.06641).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Pools K, V along the time axis with a fixed stride-R average *before*
    the softmax, so attention reads from R = ceil(T/R) ≈ 256 rebasins
    instead of T=2048 raw positions. The rebase is *before* the softmax
    (distinct from NSA which adds a global branch *after* attention, and
    from diff-attn which acts on the *output* of attention). The rebased
    softmax acts as a soft locality prior: each rebasin pools 8 tokens
    by default.

    `rebase_stride=8` is the default; larger strides reduce the
    compression. With `rebase_stride >= T` (e.g. `rebase_stride=2048`),
    R = 1 (or R = T) and the rebase is a no-op — the bit-identical
    case. The lever is the *stride* sweep on the standard `tiny1m3m`
    0.94M / 3M-token setup. Identity at step 0: when the flag is OFF
    (default), the MHA's rebase branch is never taken and the standard
    softmax path is bit-identical to the no-flag baseline. When the
    flag is ON but `rebase_stride >= T`, the pool collapses to a
    no-op (the causal mask at the rebased level equals the standard
    causal mask and the avg-pooled V equals the un-pooled V under
    a uniform-time identity).

    PASS ≤ ctrl − 0.005 (small/null band — taste puts leverage at the
    low end for an attention-side locality prior at 12L depth, where
    the existing SWA levers (closed) and the rebased-softmax neighbor
    are already in the explored space). NULL band |Δ| < 0.005. DRIFT
    > +0.005 (the avg-pool destroys per-token K/V resolution). See
    `autoresearch/ideas/154-rebased-attn/idea.md`.
    """
    use_rebased_attn: bool = True
    rebase_stride: int = 8


@dataclass
class Tiny1M3MMoAConfig(Tiny1M3MConfig):
    """Tiny1M3M with Mixture-of-Attentions (MoA) per layer.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Each block's MHA runs `E = moa_num_experts` (default 2) parallel
    attention computations with SEPARATE K_e, V_e projections (Q is
    shared across experts), then mixes the E attention outputs by a
    per-token router `g_e = softmax(W_g x)_e`. Expert 0 reuses the
    standard W_K, W_V slices of the merged qkvo_proj so the bit-
    identical init claim holds against the no-flag baseline. At init
    the (E-1) extra K/V projections are zero (extra experts produce 0)
    and the router bias is one-hot on expert 0 (g_0 = 1.0) ⇒ step-0
    output is bit-identical to a single standard attention.

    Distinct from MoS (144, closed) which mixes softmax *variants*
    within one attention — MoA mixes full attention computations.
    Distinct from the FFN-side MoE levers (117-soft-moe, 118-MoD,
    145-expert-choice, 146-sparse-ffn, all closed) which route on the
    FFN side — MoA routes on the attention side. The lever asks: is
    the *attention output* the binding capacity constraint at 0.94M
    (in which case multi-attention-per-layer helps) or is it the FFN
    / residual stream (in which case MoA nulls like the FFN MoEs did).

    Cost when on (E=2): (E-1)·(2·kv_size·d_model) extra K/V + d_model·E
    router params per layer ≈ 4-5K params/layer at tiny1m3m (~5% of
    the 0.94M model, ~50,688 extra params total). Uses one fused
    SDPA call over the combined (B·E) batch dim ⇒ no extra kernel
    launches and no score-side modifications (composes with the
    no-auxiliary-loss routing of 145-expert-choice). NULL band |Δ| ≤
    0.01. DRIFT > +0.01 (the per-token router training is a slow
    signal at 92 update steps, like 117-118-145-146). PASS ≤ −0.01.
    See `autoresearch/ideas/156-moa/idea.md`.
    """
    use_moa: bool = True
    moa_num_experts: int = 2


# =====================================================================


@dataclass
class Tiny1M3MGAUConfig(Tiny1M3MConfig):
    """Tiny1M3M with the Gated Attention Unit (Hua et al. 2022,
    arXiv:2202.10447).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    Replaces every `TransformerBlock` in the stack with a single
    `GAUBlock` — a fused Attention + FFN unit that merges the FFN's
    gating and the attention's gating into ONE block via a shared
    gating pair `(U_g, V_o)` plus a 5-projection fused linear
    `(Q, K, V, U_g, V_o)` and a separate output projection `U_o`.

    Step-0 identity (the spec pin):
        U_g = 0  →  y = x
        V_o = 0  →  V_o · z = 0  →  U_o(0) = 0  →  block(x) = x
    ⇒ at step 0 every block is the identity, so the residual stream
    passes through unchanged. The `transformer_blocks` stack is NOT
    built when `use_gau=True` (the GAU block replaces it entirely);
    the model loop dispatches via `gau_blocks[i]`.

    Param cost at tiny1m3m (d_model=64, n_heads=4, d_k=16, n_kv_heads=2,
    kv_size=32, d_ff=256, 12 layers):
        TransformerBlock per layer (squared_relu FFN, GQA):
            qkvo:  (64 + 2·32 + 64)·64  = 12,288
            FFN :  2·64·256              = 32,768
            Total                         ≈ 45K (≈558K total over 12)
        GAUBlock per layer:
            fused: (64 + 2·32 + 2·64)·64 = 12,288  (Q,K,V,U_g,V_o)
            out  : 64·64                  =  4,096  (U_o)
            norm : 64 (gain only)          ≈    64
            Total                         ≈ 16K (≈196K total over 12)
        Saving: ~29K/layer × 12 ≈ 350K (~37% of the 0.94M model).
    The freed budget can be re-spent on attention dim per the GAU
    paper's retuning — out of scope for this 1-idea A/B at fixed
    tiny1m3m tier; the model is simply smaller, which we report on.

    Distinct from every closed lever at this tier (the FFN MoEs 117,
    118, 145, 146 all close; the FFN-internal conv 157 closed; the
    pre-attn conv 143 closed null). The GAU lever is *not* an FFN-
    internal modification — it ELIMINATES the FFN entirely and folds
    the FFN's role into the attention block's gating pair. Tests
    whether the separation between Attention and FFN is the binding
    structural bottleneck at 0.94M (a WIN would suggest it is) or
    whether the per-block computation budget is what matters and the
    smaller GAU model simply under-trains (a NULL or DRIFT).

    Mutual-exclusions (asserted at construction):
        - `use_yoco`: GAU has no MHA + shared-KV; YOCO needs both.
        - `use_hyper_connections`: wrapper assumes standard block.
        - `use_value_residual`: stash lives on `block.attention`.
    None of these are enabled at the tiny1m3m baseline, so the
    default config combination (no flags) is compatible with GAU.

    Transfer-risk: low. GAU was tested by Google at T5 scale (250M-
    13B) and the bit-budget-free FFN savings is qualitatively
    different from all the closed MoE/conv levers — if GAU wins at
    0.94M, the transfer risk is mild because GAU's design point
    *is* the small-scale regime (it specifically targets parameter
    efficiency at low budgets).

    NULL band |Δ| ≤ 0.01. DRIFT > +0.01. PASS ≤ −0.01. See
    `autoresearch/ideas/158-gau/idea.md`.
    """
    use_gau: bool = True


@dataclass
class Tiny1M3MQOnlyNormConfig(Tiny1M3MConfig):
    """Tiny1M3M with Q-only RMSNorm (162 — asymmetric QK pre-softmax).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Apply
    `nn.RMSNorm(d_head, eps=1e-6)` to Q only, leave K untouched, before
    the QK matmul. nn.RMSNorm weight=1, bias=0 init ⇒ at step 0 the
    lever rescales Q to unit RMS per head-dim (spec-allowed fp32
    max-abs-diff < 1e-3 tolerance, same trade-off as 159-emb-layernorm).
    Default off ⇒ baseline path bit-identical (no `q_only_norm`
    module is registered, no forward branch taken).

    The orthogonal ablation to 016-qk-norm (which norms BOTH Q and K).
    Sharp 3-way test (with the K-only and QK-symmetric levers):
    WIN ⇒ Q-side normalization is the binding axis; NULL ⇒ K-side
    or symmetry was carrying 016's gain. Either outcome closes the
    QK-norm-attribution axis at 0.94M.

    Transfer-risk: low (RMSNorm family production-validated at LLaMA 3
    / Qwen 2.5 / Mistral 1B+; Cohere Command-R validates asymmetric QK
    at 35B+). See `autoresearch/ideas/162-q-only-norm/idea.md`.

    @dataclass-decorated so `use_q_only_norm` default is properly
    overridden (the dataclass-inheritance pitfall documented in
    `_arq_161-dyt-temp.py`).
    """
    use_q_only_norm: bool = True


@dataclass
class Tiny1M3MQCarryConfig(Tiny1M3MConfig):
    """Tiny1M3M with Cross-Block Q Residual ("Q-Carry", 164).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
    For each block l >= 1, augment the Q projection with a learnable
    α_l-scaled carry from the previous block's MHA sublayer input
    (LN(x_{l-1}), `.detach()`-ed):
        `Q_l = W_Q(x_l) + α_l · W_Q(prev_x)`,
    where `α_l = nn.Parameter(torch.zeros(()))` is a per-block 0-dim
    scalar (init 0 ⇒ step-0 forward is bit-identical to no-carry
    baseline within fp32 rounding noise of one extra multiply-add).
    The stash is set on layer 0 (no previous block exists) and read
    back by the model loop for layers 1..N-1.

    Q-side dual of 021-value-residual (which carries V): tests
    whether 021's WIN was V-specific or generalizes to "cross-block
    residual-stream mixing." K, V, O projections are unchanged; the
    carry is added before q_norm / RoPE so 016 and 162 still
    rescale `Q + α·Q_carry` consistently. The Q projection of
    `q_carry` uses the SAME W_Q slice that produced the current Q
    (default/shared_kv/MLA: `qkvo_proj[:q_size]`; tied-QK:
    `qk_proj[:q_size]`).

    Cost: 12 scalars total (one per block, +0.001% of 0.94M). FLOPs:
    +1 W_Q matmul per block, ~+11% per-step at tiny1m3m.

    @dataclass-decorated so `use_q_carry` default is properly
    overridden (the dataclass-inheritance pitfall documented in
    `_arq_161-dyt-temp.py`).
    """
    use_q_carry: bool = True
