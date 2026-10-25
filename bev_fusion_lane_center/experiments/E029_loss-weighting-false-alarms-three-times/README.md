# E029_loss-weighting-false-alarms-three-times

**Loss weighting: false alarms three times costlier than misses**

| | |
|---|---|
| experiment id | E29 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loops02-03 |

## Result

moves the threshold only, F1 change at most 0.003

## Techniques used (register K ids)

- K62: U-Net / FCN dense head  (classic)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `e29_lam3`, `e29_miss3`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
