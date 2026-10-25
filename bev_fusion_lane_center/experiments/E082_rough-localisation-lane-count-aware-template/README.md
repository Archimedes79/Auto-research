# E082_rough-localisation-lane-count-aware-template

**Rough localisation with a lane-count-aware template (ICP), held through a failure**

| | |
|---|---|
| experiment id | E82 |
| theme | T4 - Perception-first graph with a learned assignment (reference system A) (loops 8 - 9) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB04 correlation / cost volume + soft-argmax |
| loop | loop08 - log: [../../experiment_execution/indexes/LOOP8_INDEX.md](../../experiment_execution/indexes/LOOP8_INDEX.md) |

## Result

pose error 1.11 to 0.62 m; route success at most +0.016

## Techniques used (register K ids)

- K99: differentiable localisation of a BEV raster against a navigation map (pose volume, soft-argmax)  (checked loop 8)
- K100: localisation to SD maps: regression, segmentation matching, homography  (checked loop 8 (abstract / html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 8 small files from `e82_geo_clean`, `e82_geo_clean3`, `e82_geo_pert`, `e82_lanewidth`, `e82_loc`, `e82_locstats`, `e82_trackloc`, `e82_trackloc_dbg`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
