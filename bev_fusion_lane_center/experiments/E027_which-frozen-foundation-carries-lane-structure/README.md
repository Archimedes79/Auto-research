# E027_which-frozen-foundation-carries-lane-structure

**Which frozen foundation carries the lane structure: detection-pretrained CRN against map-pretrained GaussianCaR (same 2.7 M U-Net head)**

| | |
|---|---|
| experiment id | E27 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB06 view lifting (LSS, BEVFormer, splatting) |
| loop | loops02-03 |

## Result

F1 0.624 to 0.867 (VAL), 0.598 to 0.835 (TEST): +0.24, the largest single effect

## Techniques used (register K ids)

- K60: frozen camera-radar BEV features, detection-pretrained  (loop 1 foundation)
- K61: frozen camera-radar BEV features, map-pretrained (Gaussian camera branch + point-transformer radar branch)  (loops 2 - 15 foundation)
- K67: fuser and decoder of the foundation: cross-modal fusion, dense prediction transformer, point transformer v3  (part of K61)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 10 small files from `e27_crn`, `e27_gcar`
- `code/`: `make_e27_e30.py` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
