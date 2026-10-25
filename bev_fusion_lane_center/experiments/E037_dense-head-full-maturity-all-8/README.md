# E037_dense-head-full-maturity-all-8

**Dense head at full maturity: all 8,414 rows, stabilised recipe, cross-fitted heads**

| | |
|---|---|
| experiment id | E37 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB01 convolution, U-Net / FPN pyramids |
| loop | loops02-03 |

## Result

F1 0.959 / 0.885 / 0.856 (train-eval / VAL / TEST)

## Techniques used (register K ids)

- K61: frozen camera-radar BEV features, map-pretrained (Gaussian camera branch + point-transformer radar branch)  (loops 2 - 15 foundation)
- K62: U-Net / FCN dense head  (classic)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 35 small files from `e37_ens`, `e37_final`, `e37_final2_attempt_bf16_nan_log.txt`, `e37_final_attempt1_lr2e-3_log.txt`, `e37_final_attempt2_lr1e-3_log.txt`, `e37_final_attempt3_stale_log.txt`, `e37_fold0`, `e37_fold0_attempt1_lr1e-3_log.txt` ...
- `code/`: `make_e37.py`, `run_e37_ens.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
