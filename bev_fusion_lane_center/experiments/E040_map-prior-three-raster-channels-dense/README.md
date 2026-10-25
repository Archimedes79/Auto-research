# E040_map-prior-three-raster-channels-dense

**Map prior as three raster channels in the dense head (no-map control, pose noise, outdated map)**

| | |
|---|---|
| experiment id | E40 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB17 conditioning and gating |
| loop | loops02-03 |

## Result

exact pose +0.075 F1; 1 m error turns it into -0.057 unless the map is aligned first

## Techniques used (register K ids)

- K73: SD map as raster canvas and vector tokens; noise-trained against shifted priors  (checked loop 7 (html))
- K52: SD / satellite map prior for topology  (checked (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 55 small files from `e40_clean`, `e40_clean_s03_align`, `e40_clean_s03_noalign`, `e40_clean_s05_align`, `e40_clean_s05_noalign`, `e40_clean_s1_align`, `e40_clean_s1_noalign`, `e40_clean_s2_align` ...
- `code/`: `_e40_base.py`, `make_e40.py`, `run_e40_e41.sh`, `run_e40_extra.sh`, `run_e40_na.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
