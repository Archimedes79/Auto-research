# E028_which-feature-level-foundation-read

**Which feature level of the foundation to read**

| | |
|---|---|
| experiment id | E28 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB01 convolution, U-Net / FPN pyramids |
| loop | loops02-03 |

## Result

decoder feature 0.874 / 0.840 (VAL / TEST); fused BEV 0.772 / 0.743; map logits 0.779 / 0.738

## Techniques used (register K ids)

- K61: frozen camera-radar BEV features, map-pretrained (Gaussian camera branch + point-transformer radar branch)  (loops 2 - 15 foundation)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 6 small files from `e28_dec32`, `e28_fused32`, `e28_logit7`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
