# E030_train-test-gap-dense-head-data

**Train-to-test gap of the dense head: data amount, scenes, regularisation, mirroring**

| | |
|---|---|
| experiment id | E30 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB01 convolution, U-Net / FPN pyramids |
| loop | loops02-03 |

## Result

no under-fitting; 0.033 head over-fitting plus a seen-to-unseen scene shift of 0.048 (VAL) / 0.079 (TEST)

## Techniques used (register K ids)

- K62: U-Net / FCN dense head  (classic)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 23 small files from `e30_base2`, `e30_boot.json`, `e30_boot_counts.json`, `e30_boot_loc.json`, `e30_flip`, `e30_fold_seen.json`, `e30_reg`, `e30_s16` ...
- `code/`: `make_e30.py`, `run_e30_adj.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
