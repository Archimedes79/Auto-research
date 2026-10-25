# E079_ego-motion-warped-average-lane-output

**Ego-motion-warped average of the lane output; low-error map-free hypotheses**

| | |
|---|---|
| experiment id | E79 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB10 memory bank / query propagation |
| loop | loop07 |

## Result

+0.042 lane F1 in a failure phase (no training)

## Techniques used (register K ids)

- K78: global raster memory accumulated from past predictions  (checked loop 7 (html))
- K102: decayed, warped evidence; temporal consistency loss  (checked loop 8 (html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 20 small files from `e79_range`, `e79_range_nt`, `e79_test`, `e79_test_align_odo0`, `e79_test_align_odo1`, `e79_test_nt`, `e79_test_nt_mapfree`, `e79_test_nt_odo03` ...
- `figures/`: 1 images from the report sources
- `code/`: `analyze_e79.py`, `fig_e79.py` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
