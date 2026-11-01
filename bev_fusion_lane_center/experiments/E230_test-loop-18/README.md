# E230_test-loop-18

**TEST (once): pruned champion f1, j1, control with the route head along the edges (original and failure-trained heads)**

| | |
|---|---|
| experiment id | E230 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | C - classical / reference / analysis |
| loop | loop18 - state: [../../experiment_execution/history/indexes/LOOP18_STATE.md](../../experiment_execution/history/indexes/LOOP18_STATE.md) |

## Result

TEST (833 frames): f1 + pruning S 0.748, Q 0.718, J 0.422, lanes found 0.653, route 0.841 / 0.844 (failure-trained 0.859 / 0.865); j1 + pruning S 0.746, J 0.419; rc1 + pruning 0.4 S 0.747, Q 0.727, J 0.463, lanes found 0.724; control route 0.894 / 0.868 (loop-16 TEST S 0.702, J 0.404, lanes found 0.762). Details: evaluation/TEST_USAGE_LOG.md.

## Data

- `results/test18_table.json`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `code/` (all loop-18 scripts in their original layout: git tag `lcr/loop18-scratch`)
- raw outputs (arrays, checkpoints): `../../runs/`
