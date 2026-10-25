# E104_cross-attention-from-vertex-tokens-navigation

**Cross-attention from vertex tokens to navigation-map tokens with gated fusion (0.69 M)**

| | |
|---|---|
| experiment id | E104 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB12 cross-attention; BB17 conditioning and gating |
| loop | loop10 |

## Result

route success 0.853 / 0.843 / 0.832 / 0.820 at 0 / 1 / 2 / 3 m, 0.007 - 0.026 below the registration-based network

## Techniques used (register K ids)

- K75: assignment of perceived lanes to SD roads (path-aware attention, beam search)  (checked loops 7, 14)
- K98: prior fusion with log-confidence bias, presence gate and source dropout  (checked loop 8 (html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 32 small files from `e104_smoke`, `e104a`, `e104b`, `e104b_s1`, `e104br`, `e104br_s1`, `e104bz`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
