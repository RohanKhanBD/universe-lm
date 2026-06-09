# Evidence — 002 cautious-adamw

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast-34386 (220.82.52.202:34386, RTX 3060, sm_86)
- control val: 6.4403
- treatment A val: 6.4406  (embedding bucket: `token_embedding` + `emb_proj`)  Δ: +0.0003
- treatment B val: 6.4337  (gain bucket: `*.norm.weight` + 1D scalars)            Δ: -0.0066
- pass/fail bar: idea plan calls for screen20m tier (val ≤ 4.6314, Δ ≤ -0.005); tiny1m3m not resolvable per plan, expected noise ±0.06-0.16 → both A and B in noise
- → not met (NULL — by design at this tier)
- box check: ctrl 6.4403 vs leaderboard 6.4287 (+0.0116, within noise; BOX OK)
- raw: remote-results/2026-06-09-vast-tiny1m3m/results.json
- date: 2026-06-09

## Re-test on resolvable tier
The idea's plan requires screen20m (V+q+SWA+HighRoPE 4.6364 control, target Δ ≤ -0.005).
At screen20m the per-bucket A/B becomes resolvable. The 002 wiring (`use_cautious_adamw`
flag + `CautiousAdamW` subclass) is in place and bit-identical when `"none"` (default);
`boxval` smoke (max diff 2.98e-08) confirms the gate. To re-test, run on screen20m tier.
