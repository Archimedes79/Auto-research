# E217_far-range-recall-negatives

**Far-range recall gap to the control: beam budget, per-level threshold, validity sweep, candidate coverage**

| | |
|---|---|
| experiment id | E217 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | D - dropped: tried, no gain or worse |
| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |

## Result

58 % of the recall gap lies beyond 40 m (recall 0.709 vs 0.811); beams 400 / 200 and 576 / 288: lanes found 0.700 / 0.701; the tree has two levels (per-level threshold: 0.707); tau 0.2: R@1 0.858; all valid nodes cover 85 % of the far points, the admitted 74 %. Not a budget or threshold effect.

## Data

- no separate data files (numbers in the loop-18 report and the state file)
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `../../temp/launchers/loop18/`
- raw outputs (arrays, checkpoints): `../../runs/`
