# E015_frozen-crn-camera-radar-features-nuscenes

**Frozen CRN camera-radar features on nuScenes Mini: layout, feature source, channel count, decoder changes**

| | |
|---|---|
| experiment id | E15 - E16 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB06 view lifting (LSS, BEVFormer, splatting) |
| loop | loop01 |

## Result

heat-map F1 0.58; mirrored layout 0.37 - 0.43; channel count and level irrelevant (0.55 - 0.58)

## Techniques used (register K ids)

- K60: frozen camera-radar BEV features, detection-pretrained  (loop 1 foundation)
- K66: dataset  (dataset)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
