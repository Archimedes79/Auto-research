# E038_ceiling-grid-proposals-cascade-refiner

**Ceiling of the grid proposals; cascade refiner (3 blocks, pooling + re-estimation)**

| | |
|---|---|
| experiment id | E38 / E39 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP2 line forming (dense field to lines: threshold, thinning, tracer); SP6 vertex positions of the sparse graph (refinement, keep gate) |
| building blocks | BB08 iterative refinement, cascades |
| loop | loops02-03 |

## Result

98.2 % of the ground truth within 2.5 m; refiner +0.001 F1, +0.014 route F

## Techniques used (register K ids)

- K42: cascade with increasing thresholds (= K22, now checked)  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `make_e38.py`, `make_e39.py`, `run_e38.sh`, `run_e39.sh`, `run_e39reg.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
