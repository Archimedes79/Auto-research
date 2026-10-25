# E220_failure-trained-route-head-in-the-best-chain

**Route heads trained on failure frames in the best chain (builder B + gate, exact map, fork pruning)**

| | |
|---|---|
| experiment id | E220 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |

## Result

Failure half of the best chain, route success original heads s0 / s1 -> failure-trained c0 / c1: night 0.832 / 0.818 -> 0.847 / 0.847, blur 0.883 / 0.854 -> 0.905 / 0.891, three cameras off 0.774 / 0.825 -> 0.832 / 0.847. Rule (>= +0.10) not met: the map builder already carries the route.

## Techniques used (register K ids)

- K190 (see `../../literature/register.md`)

## Data

- `results/e220_table.txt`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `../../temp/launchers/loop18/`
- raw outputs (arrays, checkpoints): `../../runs/`
