# Bridge Early Layers to Late Layers to Improve Your LLM Training: U-Net Skips

By Vuk Rosić

We added this trick to a small model and validation loss improved:

![val loss curves](images/01_val_loss_curves.png)

A U-Net skip connects an early layer in a transformer to a matching late layer.

It is just a small learned bridge that the model can scale up or down.

The bridge starts almost off, so it never hurts the model at the start of training.

Late layers can reach back to the simple, local features that the early layers already saw.

It also gives gradients a shorter path back to the early layers.

That shorter path is what helps deep models train.

![u-net skip architecture](images/unet_architecture.png)

## How it works, step by step

A deep transformer processes tokens one layer at a time.

Each layer only reads the output of the layer right below it.

Early layers tend to capture simple, local patterns.

By the time information reaches the late layers, those early details can get washed out.

A U-Net skip fixes this by saving the output of each early layer.

It then adds each saved output back into a matching late layer.

The pairing is symmetric, like the two sides of a letter U.

The first layer connects to the last, the second to the second-to-last, and so on.

```text
layer 0  ->  layer 7
layer 1  ->  layer 6
layer 2  ->  layer 5
layer 3  ->  layer 4
```

Each bridge is gated, so the model decides how much of the early output to pull in.

The gate runs through a sigmoid.

The sigmoid keeps the gate strength between 0 and 1.

That makes the gate stable to train.

![the sigmoid gate scales the skip before adding it](images/unet_gate.png)

The gate weight starts at -1.5, so `sigmoid(-1.5)` is about 0.18.

The skip starts small but not exactly zero.

A small nonzero start matters: a gate that starts at exactly zero gets almost no gradient.

It can fail to ever turn on.

So the skip begins faint, and training raises or lowers it per dimension as the model learns how useful the bridge is.

## In code

Keep one gate vector per bridge, initialized to -1.5.

```python
# n_skips bridges, one gate value per embedding dimension
gate = nn.Parameter(torch.full((n_skips, d_model), -1.5))
# sigmoid(-1.5) ~ 0.18  ->  the skip starts small but nonzero
```

In the first half of the layers, save each layer's output.

```python
skips = []
for i, block in enumerate(blocks):
    x = block(x)
    if i < n_skips:
        skips.append(x)        # remember early outputs
```

In the second half, before each layer, add its matching early output through the sigmoid gate.

```python
for i, block in enumerate(blocks):
    if i >= n_layers - n_skips:
        j = n_layers - 1 - i               # matching early layer
        x = x + torch.sigmoid(gate[j]) * skips[j]
    x = block(x)
```

Put together, the first half writes its outputs.

The second half reads them back, scaled by a learned per-dimension gate.

This is the same skip pattern used in the record-setting nanoGPT speedrun, where the sigmoid gate and the small nonzero start are what make it train well.

## Phase 1 ablation - does the bound help, and does skip count matter?

Eight runs on a tiny ~1M-param model (12 layers, d_model 64), trained for 733 steps (~3M tokens), all with seed 42.

The table below is the source of truth.

The three "raw0" rows collapsed to one in the table are bit-identical runs (the skip-count flag was ignored for them, so they all used the default k=6).

### Runs

| run | use_unet_skips | gate_type | gate_init | skip_count | val_loss | val_ppl |
|---|---|---|---|---|---|---|
| `tiny_unet_ctrl` | false | - | - | 0 | 6.4422 | 627.78 |
| `tiny_unet_raw0_k2_real` | true | raw | 0.0 | **2** | 6.4503 | 632.90 |
| `tiny_unet_raw0` | true | raw | 0.0 | 6 (default) | 6.4203 | 614.20 |
| `tiny_unet_raw018` | true | raw | 0.18 | 6 (default) | 6.4272 | 618.43 |
| `tiny_unet_sigmoid_m15` | true | sigmoid | -1.5 | 6 (default) | **6.3816** | **590.85** |
| `tiny_unet_sigmoid_m30` | true | sigmoid | -3.0 | 6 (default) | 6.3831 | 591.77 |

![final val_loss bar chart - sigmoid wins by 0.04-0.07](images/02_final_loss_bar.png)

### What the data shows

The sigmoid bound beats raw scalar gates by 0.04-0.07 val_loss.

The best run (`sigmoid_m15`, 6.3816) is 0.039 below the best raw run.

It is 0.061 below the no-skip control.

The benefit is robust to the initial value.

`sigmoid_m15` and `sigmoid_m30` differ by only 0.0015 val_loss.

That is well inside run-to-run noise.

![sigmoid vs raw - same init, sigmoid ends lower](images/03_sigmoid_vs_raw.png)

The sigmoid advantage comes from the [0, 1] bound, not the start point.

`sigmoid(-1.5)` is about 0.18 - the same effective skip weight as `raw_init=0.18` at initialization.

The two runs start with identical skip signal strength, but sigmoid ends 0.046 val_loss better.

The bound prevents the gate from drifting to large values that would let the skip dominate the residual stream.

![skip count sweep - only one k=2 point is real (see caveats)](images/04_skip_count_scatter.png)

Skip count is non-monotonic for raw gates.

With raw gates at init 0, k=2 (only the deepest two bridges) is *worse* than no skips at all.

k=6 (the full U) is *better* than no skips.

The model needs a minimum bridge count to overcome the dead-start of raw gates at 0.

Fewer than that minimum is a net loss.

### Caveats

The skip_count sweep is incomplete.

Of the original 8 runs, only `raw0_k2_real` actually used a non-default `skip_count`.

The `raw0`, `raw0_k2`, and `raw0_k4` names were cosmetic.

The launch script never passed the skip-count flag for those three, so all three used the default k=6 and produced bit-identical val_loss curves.

A re-run with the flag actually passed is needed to map out k=1, 2, 4, 5 for both families.

The `raw0_k2_real` run used a different code path from the others (the "real" branch in the original training script's history).

Whether the k=2 dip is a property of the architecture or an artifact of that branch is unresolved.

All runs are tiny1m at 3M tokens.

Scaling up is the obvious next test.

If the sigmoid-vs-raw gap holds or grows at a 5-10M model, the bound is doing real work and not just a small-model artifact.

### Want to go deeper in AI research? I'll coach you 1-on-1

Bring whatever you're stuck on - picking a direction, your first experiment, a paper you can't crack, your training setup, or a career move.

📆 **$20 (80% OFF) for the founding cohort - first 8 spots.** Not a fit? I'll
refund in the first 10 minutes, no hard feelings.
→ https://cal.com/vuk-ai/60-min

### Not ready for a call? Start free in the Skool

Every experiment I post comes with the scaffolded code and a step-by-step
protocol, so you can reproduce it yourself and then run your own variant. You also get the weekly research thread and a community of people doing real AI research.
Free to try.
→ https://www.skool.com/become-ai-researcher-2669/about
