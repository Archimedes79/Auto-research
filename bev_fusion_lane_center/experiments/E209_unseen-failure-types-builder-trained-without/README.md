# E209_unseen-failure-types-builder-trained-without

**Unseen failure types: map builder and its no-map control trained WITHOUT night and blur rows, evaluated on night and blur (rerun with checkpoints)**

| | |
|---|---|
| experiment id | E209 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/history/indexes/LOOP18_STATE.md](../../experiment_execution/history/indexes/LOOP18_STATE.md) |

## Result

Failure half, exact map, paired scene bootstrap of F1@1: single frame night S 0.504 / F1 0.465 (no-map control 0.454 / 0.373, champion 0.419 / 0.312, builder that saw night 0.568 / 0.619), map +0.093 [+0.065, +0.122]; blur single 0.577 / 0.617 (control 0.530 / 0.531), +0.086 [+0.053, +0.125]; with the spatial hold night 0.655 / 0.749 (control 0.635 / 0.711, seen 0.662 / 0.780), +0.038 [+0.024, +0.057]; blur 0.661 / 0.773 (control 0.631 / 0.723, seen 0.660 / 0.782), +0.050 [+0.031, +0.070]. The map benefit transfers to unseen failure types.

## Data

- `results/e209.md`
- `figures/p18_e209_unseen.png`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `code/` (all loop-18 scripts in their original layout: git tag `lcr/loop18-scratch`)
- raw outputs (arrays, checkpoints): `../../runs/`
