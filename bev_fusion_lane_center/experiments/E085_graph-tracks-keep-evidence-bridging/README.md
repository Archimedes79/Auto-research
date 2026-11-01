# E085_graph-tracks-keep-evidence-bridging

**Graph tracks with keep evidence and bridging (temporal fusion on the graph)**

| | |
|---|---|
| experiment id | E85 |
| theme | T4 - Perception-first graph with a learned assignment (reference system A) (loops 8 - 9) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB10 memory bank / query propagation |
| loop | loop08 - log: [../../experiment_execution/history/indexes/LOOP8_INDEX.md](../../experiment_execution/history/indexes/LOOP8_INDEX.md) |

## Result

+0.110 failure-phase lane F1 (night +0.35); needs odometry within 1 degree per sqrt(s)

## Techniques used (register K ids)

- K101: global vector map with association and map NMS  (checked loop 8 (html))
- K102: decayed, warped evidence; temporal consistency loss  (checked loop 8 (html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 9 small files from `e85_a`, `e85_dev`, `e85_early`, `e85_early_a5`, `e85_early_a85`, `e85_early_d85`, `e85_ens`, `e85_ensR` ...
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
