# E061_temporal-tracking-map-alignment-lane-id

**Temporal tracking of the map alignment and of the lane-ID assignment**

| | |
|---|---|
| experiment id | E61 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB10 memory bank / query propagation |
| loop | loop05 |

## Result

route success stays 0.92 where per-frame registration falls 0.845 to 0.608

## Data

- `results/`: 3 small files from `e61_s1`, `e61_s2`, `e61_s3`
- code: removed in the clean-up of 2026-11-21, in the git history (commit `57b9eab1`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
