# E221_calibration-of-fork-heat-and-node-validity

**Calibration of the junction-head heat at predicted forks and of the node validity (literature gap 3)**

| | |
|---|---|
| experiment id | E221 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | C - classical / reference / analysis |
| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |

## Result

Fork heat is calibrated (ECE 0.034 split / 0.037 merge); node validity is bimodal and not calibrated in the middle (ECE 0.121 raw, 0.021 after an isotonic fit).

## Data

- no separate data files (numbers in the loop-18 report and the state file)
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `code/` (all loop-18 scripts in their original layout: git tag `lcr/loop18-scratch`)
- raw outputs (arrays, checkpoints): `../../runs/`
