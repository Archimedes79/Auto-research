# E023_ensemble-decoder-families-coordinate-channels-distance

**Ensemble of decoder families; coordinate channels and a distance-dependent threshold**

| | |
|---|---|
| experiment id | E23 / E25 |
| theme | T1 - Foundation and dense lane field (loops 1 - 4) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB18 position codes |
| loop | loop01 |

## Result

ensemble 0.650 against 0.670; coordinate channels not confirmed with full training

## Techniques used (register K ids)

- K62: U-Net / FCN dense head  (classic)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `chain_e25.sh`, `run_crn_e25.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
