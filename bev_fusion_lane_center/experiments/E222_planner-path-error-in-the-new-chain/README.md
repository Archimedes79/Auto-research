# E222_planner-path-error-in-the-new-chain

**Planner consumer path error at 20 m in the best chain with and without fork pruning and with the failure-trained route head**

| | |
|---|---|
| experiment id | E222 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/history/indexes/LOOP18_STATE.md](../../experiment_execution/history/indexes/LOOP18_STATE.md) |

## Result

Failure half, exact map: path error at 20 m 0.28 / 0.27 / 0.28 m (night / blur / three cameras off) without pruning, 0.28 / 0.27 / 0.27 with pruning, 0.28 / 0.27 / 0.26 with pruning and the failure-trained head: no measurable planner-level effect (floor 0.27 m).

## Data

- no separate data files (numbers in the loop-18 report and the state file)
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `code/` (all loop-18 scripts in their original layout: git tag `lcr/loop18-scratch`)
- raw outputs (arrays, checkpoints): `../../runs/`
