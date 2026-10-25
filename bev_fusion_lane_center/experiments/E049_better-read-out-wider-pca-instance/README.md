# E049_better-read-out-wider-pca-instance

**Better read-out: wider PCA (96), instance normalisation, both foundations fused**

| | |
|---|---|
| experiment id | E49 / E50 / E52 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB17 conditioning and gating |
| loop | loop04 |

## Result

no gain (at most 0.008)

## Techniques used (register K ids)

- K60: frozen camera-radar BEV features, detection-pretrained  (loop 1 foundation)
- K61: frozen camera-radar BEV features, map-pretrained (Gaussian camera branch + point-transformer radar branch)  (loops 2 - 15 foundation)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `make_e49_e50.py`, `run_e49.sh`, `run_e52.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
