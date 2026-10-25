# E218_fork-pruning-under-failure-and-best-chain

**Fork pruning under sensor failure and in the best chain (map builder + gate + registration + trust)**

| | |
|---|---|
| experiment id | E218 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |

## Result

Clean half S 0.752 -> 0.780 (hold), J 0.227 -> 0.384; failure half of the best chain S 0.664 -> 0.682 (night, registered, trust 0.75), J 0.158 -> 0.247; single-frame failure S does not improve (fragments add vertices).

## Techniques used (register K ids)

- K190 (see `../../literature/register.md`)

## Data

- `results/e218.md`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `../../temp/launchers/loop18/`
- raw outputs (arrays, checkpoints): `../../runs/`
