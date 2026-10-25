# E216_numba-union-kernels

**Numba kernels for the path union of the assembly (identical graphs)**

| | |
|---|---|
| experiment id | E216 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |

## Result

873 / 873 validation graphs bit-identical to the loop-16b fast path; about 2x faster under load; idle timing in the loop-18 report.

## Techniques used (register K ids)

- K191 (see `../../literature/register.md`)

## Data

- no separate data files (numbers in the loop-18 report and the state file)
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `code/` (all loop-18 scripts in their original layout: git tag `lcr/loop18-scratch`)
- raw outputs (arrays, checkpoints): `../../runs/`
