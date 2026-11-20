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
- code: removed in the clean-up of 2026-11-21, kept behind the git tag `archive/pre-cleanup-2026-11-20`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
