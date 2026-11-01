# E112_spatial-hold-per-pixel-maximum-frame

**Spatial hold: per-pixel maximum of the frame and the decayed, warped memory (no monitor)**

| | |
|---|---|
| experiment id | E112 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB10 memory bank / query propagation |
| loop | loop12 - log: [../../experiment_execution/history/indexes/LOOP12_INDEX.md](../../experiment_execution/history/indexes/LOOP12_INDEX.md) |

## Result

night +0.34, blur +0.20 over the single frame; not for lost front cameras

## Techniques used (register K ids)

- K102: decayed, warped evidence; temporal consistency loss  (checked loop 8 (html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 7 small files from `e112_a`, `e112_ramp8`
- `code/`: `_e112.sh`, `_e112b.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
