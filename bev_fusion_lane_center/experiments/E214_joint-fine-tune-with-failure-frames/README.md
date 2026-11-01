# E214_joint-fine-tune-with-failure-frames

**Longer joint fine-tune of builder + route head with real failure frames (arm a: route loss reaches the builder; arm c: only the route head learns), two route-head seeds**

| | |
|---|---|
| experiment id | E214 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | D - dropped: tried, no gain or worse |
| loop | loop18 - state: [../../experiment_execution/history/indexes/LOOP18_STATE.md](../../experiment_execution/history/indexes/LOOP18_STATE.md) |

## Result

Joint training: no gain (arm a - arm c between -0.073 and +0.007 failure-half route success). Route head trained on failure frames (arm c): night 0.372 -> 0.591 / 0.562, blur 0.584 / 0.547 -> 0.781 / 0.803, three cameras off 0.423 / 0.401 -> 0.555 / 0.613; clean half 0.846 / 0.816 -> 0.875 / 0.868.

## Data

- `results/j2_table.txt`
- `figures/p18_route_failure_training.png`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `code/` (all loop-18 scripts in their original layout: git tag `lcr/loop18-scratch`)
- raw outputs (arrays, checkpoints): `../../runs/`
