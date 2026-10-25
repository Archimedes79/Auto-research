# E231_recall-variant-under-sensor-failure

**The j1 family (j1, equal-budget controls, recall variants j1rc) on the loop-17 failure series**

| | |
|---|---|
| experiment id | E231 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |

## Result

Failure half, j1rc - equal-budget control j1eq with the spatial hold: lanes found +0.052 / +0.058 / +0.071 (night / blur / three cameras off), Q +0.002 / -0.001 / -0.038 [-0.058, -0.020]. Rule not met (Q below the control for three cameras off): the recall weight is not uniformly the robust choice; the plain longer fine-tune is the safer fast variant where cameras can drop out.

## Techniques used (register K ids)

- K190 (see `../../literature/register.md`)

## Data

- `results/e231_table.json`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `../../temp/launchers/loop18/`
- raw outputs (arrays, checkpoints): `../../runs/`
