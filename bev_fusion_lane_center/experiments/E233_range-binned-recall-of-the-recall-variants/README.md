# E233_range-binned-recall-of-the-recall-variants

**Recall@1 m per range bin of the recall variants against their equal-budget controls and the control E94 (WP4 analysis)**

| | |
|---|---|
| experiment id | E233 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | C - classical / reference / analysis |
| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |

## Result

Beyond 40 m: j1 0.677, j1eq 0.688, j1rc 0.727, j1rc6 0.746; f1 0.709, f1eq 0.724, rc1 0.771, rc2 0.790; control 0.811. The recall weight lifts every bin, most beyond 40 m (rc1 closes 54 % and rc2 76 % of the far gap of f1eq, j1rc 32 % of j1eq's): the far gap is partly a precision-recall trade of the verifier, not only a feature limit.

## Data

- `results/range_recall.json`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `../../temp/launchers/loop18/`
- raw outputs (arrays, checkpoints): `../../runs/`
