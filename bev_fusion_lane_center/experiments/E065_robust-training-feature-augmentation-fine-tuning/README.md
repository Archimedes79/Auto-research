# E065_robust-training-feature-augmentation-fine-tuning

**Robust training: feature augmentation, fine-tuning on real perturbed features, AdaBN**

| | |
|---|---|
| experiment id | E65 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB17 conditioning and gating |
| loop | loop05 |

## Result

night 0.259 to 0.451, an unseen blur 0.433 to 0.540

## Techniques used (register K ids)

- K82: maps that stay usable with missing views or sensors: reconstruction, distillation, masked-input training, gating  (checked loops 8, 14 (abstract))
- K119: adaptive batch normalisation for test-time adaptation  (from memory, no link)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 58 small files from `e65_base`, `e65_base7`, `e65_e65_faug_blur`, `e65_e65_faug_cam_front3_off`, `e65_e65_faug_cam_front_off`, `e65_e65_faug_dark`, `e65_e65_faug_extr3`, `e65_e65_faug_fog` ...
- `code/`: `make_e65.py`, `run_e65.sh`, `run_e65_base.sh`, `run_e65_eval.sh`, `run_e65_extract.sh`, `run_e65_extract2.sh`, `run_e65_extract3.sh`, `run_e65b.sh`, `run_e65d.sh`, `run_e65e.sh`, `run_e65f.sh`, `run_e65f_extract.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
