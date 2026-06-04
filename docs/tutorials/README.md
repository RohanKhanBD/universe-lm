# Tutorials Map

This folder holds the teachable writeups from the small-compute LLM ablation work.

The rough split is:

- `README.md` files are the main tutorial entrypoints.
- companion PDFs, translations, and figures live inside the same tutorial folder.
- compact machine-readable evidence lives in `../../results/`.
- raw run folders and checkpoints stay out of git.

| tutorial | path | status |
|---|---|---|
| Value embeddings | [`value_embeddings/README.md`](value_embeddings/README.md) | polished tutorial |
| QKV embeddings | [`qkv_embeddings/README.md`](qkv_embeddings/README.md) | polished tutorial with X/PDF companion assets |
| Embedding factorization depth | [`embedding_factorization_depth/README.md`](embedding_factorization_depth/README.md) | polished tutorial |
| QK gain | [`qk_gain/README.md`](qk_gain/README.md) | polished tutorial, figures, English PDF, Chinese version |
| Normalization | [`normalization/README.md`](normalization/README.md) | active tutorial draft built from three normalization notes |

## Tutorial Folders

### QK Gain

All QK gain material stays together:

- [`qk_gain/README.md`](qk_gain/README.md) - English tutorial
- [`qk_gain/README.cn.md`](qk_gain/README.cn.md) - Chinese tutorial
- [`qk_gain/qk_gain.pdf`](qk_gain/qk_gain.pdf) - English PDF
- [`qk_gain/qk_gain_cn.pdf`](qk_gain/qk_gain_cn.pdf) - Chinese PDF
- [`qk_gain/images/`](qk_gain/images/) - source figures used by the tutorial

### Normalization

The normalization work is now in one folder so it can become one tutorial:

- [`normalization/README.md`](normalization/README.md) - main teaching draft
- [`normalization/ablations.md`](normalization/ablations.md) - short result table
- [`normalization/findings.md`](normalization/findings.md) - full findings log
