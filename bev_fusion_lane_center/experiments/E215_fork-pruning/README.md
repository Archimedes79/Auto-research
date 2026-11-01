# E215_fork-pruning

**Fork pruning: forks without junction-head support are un-linked (geometry unchanged)**

| | |
|---|---|
| experiment id | E215 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/history/indexes/LOOP18_STATE.md](../../experiment_execution/history/indexes/LOOP18_STATE.md) |

## Result

VAL 873 frames, threshold 0.3: S 0.746 -> 0.769, Q 0.701 -> 0.735, J 0.265 -> 0.409 (control 0.408), split P 0.18 -> 0.38, F1@1 0.870 and lanes found 0.695 unchanged, vertices per 100 m 13.5 -> 14.8; paired scene bootstrap S +0.024 [+0.017, +0.030], J +0.144 [+0.104, +0.182]; route success with the route head along the edges 0.894 / 0.891 -> 0.899 / 0.900; same effect on j1 192 / 96 (J 0.283 -> 0.394) and rc1 (J 0.182 -> 0.428).

## Techniques used (register K ids)

- K190 (see `../../literature/register.md`)

## Data

- `results/table.md`
- `results/results.json`
- `results/fork_features.npz`
- `results/asm_grid.log`
- `figures/p18_forkprune.png`
- `figures/p18_frames_forkprune.png`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `code/` (all loop-18 scripts in their original layout: git tag `lcr/loop18-scratch`)
- raw outputs (arrays, checkpoints): `../../runs/`
