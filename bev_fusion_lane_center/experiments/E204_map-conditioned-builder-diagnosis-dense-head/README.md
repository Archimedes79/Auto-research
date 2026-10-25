# E204_map-conditioned-builder-diagnosis-dense-head

**Map-conditioned builder (SD map raster as extra input channels), diagnosis, dense head trained on failures**

| | |
|---|---|
| experiment id | E204 / E205 / E206 |
| theme | T10 - Robustness, memory, map-conditioned builder, consumer under failure (loop 17) (loop 17) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop17 |

## Result

see the loop-17 report and experiment_execution/loop17_recovery (internal E138, E138b, E139; e17/e138_map, e139_*)

## Data

- `results/`: 13 small files from `e138b_mapdiag.json`, `e139_route`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
