# E018_cnt-net-convnext-v2-encoder-transformer

**CNT-Net: ConvNeXt-V2 encoder with a transformer bottleneck as dense decoder (4.8 M)**

| | |
|---|---|
| experiment id | E18 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB11 self-attention |
| loop | loop01 |

## Result

0.507, 0.11 below the U-Net

## Techniques used (register K ids)

- K63: ConvNeXt V2 encoder  (classic)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `chain_e18.sh`, `run_crn_e18.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
