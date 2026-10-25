# E087_training-against-map-faults-per-road

**Training against map faults: per-road bias, wrong and missing maps, abstention targets (same network as E83; the recipe carries over to system B)**

| | |
|---|---|
| experiment id | E87 |
| theme | T4 - Perception-first graph with a learned assignment (reference system A) (loops 8 - 9) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB17 conditioning and gating |
| loop | loop09 |

## Result

bias 2 m: 0.663 to 0.847; wrong-map route flag 60.7 to 4.2 m; no cost on right maps

## Techniques used (register K ids)

- K73: SD map as raster canvas and vector tokens; noise-trained against shifted priors  (checked loop 7 (html))
- K74: navigation-map polylines by cross-attention, prior treated as noise, BEV dropout  (checked loops 7, 14 (abstract))
- K98: prior fusion with log-confidence bias, presence gate and source dropout  (checked loop 8 (html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 55 small files from `e87_a`, `e87_a1`, `e87_a1_devq`, `e87_a_devq`, `e87_ab`, `e87_ab_devq`, `e87_ab_devq_tr0.2`, `e87_ab_devq_tr0.3` ...
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
