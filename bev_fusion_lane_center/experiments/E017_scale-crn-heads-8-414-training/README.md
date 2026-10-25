# E017_scale-crn-heads-8-414-training

**Scale the CRN heads to 8,414 training frames: FCN (0.1 M) against U-Net**

| | |
|---|---|
| experiment id | E17 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB01 convolution, U-Net / FPN pyramids |
| loop | loop01 |

## Result

FCN 0.264 to 0.493; U-Net +0.12 (0.613)

## Techniques used (register K ids)

- K60: frozen camera-radar BEV features, detection-pretrained  (loop 1 foundation)
- K62: U-Net / FCN dense head  (classic)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
