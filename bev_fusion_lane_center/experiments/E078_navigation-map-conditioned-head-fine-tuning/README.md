# E078_navigation-map-conditioned-head-fine-tuning

**Navigation-map conditioned head (5-channel raster) with a fine-tuning study**

| | |
|---|---|
| experiment id | E78 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB17 conditioning and gating |
| loop | loop07 |

## Result

route success 0.733 (exact) / 0.713 (2 m, registered), no map 0.328

## Techniques used (register K ids)

- K52: SD / satellite map prior for topology  (checked (abstract))
- K73: SD map as raster canvas and vector tokens; noise-trained against shifted priors  (checked loop 7 (html))
- K74: navigation-map polylines by cross-attention, prior treated as noise, BEV dropout  (checked loops 7, 14 (abstract))
- K76: spatial alignment module and sensor/prior gate  (checked loop 7 (html))
- K77: SD-map prior in lane-segment queries, denoising, past-frame consistency  (checked loops 14, 15)
- K80: tri-mode training (no prior, previous prediction, map), displaced priors  (checked loops 7, 8 (html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 44 small files from `e78_alt_final`, `e78_alt_nt`, `e78_final`, `e78_poorq_final`, `e78_poorq_nt`, `e78_sd_a`, `e78_sd_init_e60`, `e78_sd_lr1e3` ...
- `figures/`: 2 images from the report sources
- `code/`: `_insert_e78.py`, `analyze_e78.py`, `fig_e78_qual.py` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
