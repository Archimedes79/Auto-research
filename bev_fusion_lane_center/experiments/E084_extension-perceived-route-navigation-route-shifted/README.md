# E084_extension-perceived-route-navigation-route-shifted

**Extension of the perceived route with the navigation route, shifted by the perceived lane offset**

| | |
|---|---|
| experiment id | E84 |
| theme | T4 - Perception-first graph with a learned assignment (reference system A) (loops 8 - 9) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loop08 - log: [../../experiment_execution/history/indexes/LOOP8_INDEX.md](../../experiment_execution/history/indexes/LOOP8_INDEX.md) |

## Result

+0.30 (three front cameras off), +0.17 (night), +0.01 (clean)

## Techniques used (register K ids)

- K85: global nodes then local repair; extended line for occlusion (SAM-Road++ is named in the loop-8 text but its arXiv id 2411.16733 is another p  (checked loop 8 and link check 2026-10-23)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 5 small files from `e84_a`, `e84_ens`, `e84_ensQ`, `e84_ensR`, `e84_ens_tk9`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
