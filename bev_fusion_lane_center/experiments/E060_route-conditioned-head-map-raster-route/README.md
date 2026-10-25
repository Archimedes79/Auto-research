# E060_route-conditioned-head-map-raster-route

**Route-conditioned head: map raster and route in, lane and route heat maps out; systematic fine-tuning on perturbed features**

| | |
|---|---|
| experiment id | E60 / E73 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB17 conditioning and gating |
| loop | loop05 |

## Result

+0.050 lane F1 with the exact map; corrupted features carry the gain (+0.059 DEV)

## Techniques used (register K ids)

- K68: linear probing then full fine-tuning  (checked loop 6 (abstract))
- K69: surgical fine-tuning of chosen layers  (checked loop 6 (abstract))
- K70: weight-space interpolation of fine-tuned and original model  (checked loop 6 (abstract))
- K71: model soups: averaging fine-tuned weights  (checked loop 6 (abstract))
- K72: L2-SP: penalty towards the pre-trained weights  (checked loop 6 (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `analyze_e73.py`, `fig_e73.py`, `make_e60.py`, `run_e60.sh`, `run_e60b.sh`, `run_e60c.sh`, `run_e73_extract.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
