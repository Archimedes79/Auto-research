# E024_error-analysis-crn-heads-range-place

**Error analysis of the CRN heads by range and place**

| | |
|---|---|
| experiment id | E24 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loop01 |

## Result

recall 0.81 (0 - 10 m) falls to 0.50 (40 - 48 m); the loss sits in the frozen features

## Techniques used (register K ids)

- K60: frozen camera-radar BEV features, detection-pretrained  (loop 1 foundation)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
