# E077_adasis-like-navigation-map-simulator-rule

**ADASIS-like navigation map simulator and a rule-based lane assignment baseline**

| | |
|---|---|
| experiment id | E77 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loop07 |

## Result

rule assignment 0.46 against 0.93 at exact pose

## Techniques used (register K ids)

- K120: ADASIS v2 horizon specification  (checked loop 7 (specification))
- K73: SD map as raster canvas and vector tokens; noise-trained against shifted priors  (checked loop 7 (html))
- K74: navigation-map polylines by cross-attention, prior treated as noise, BEV dropout  (checked loops 7, 14 (abstract))
- K75: assignment of perceived lanes to SD roads (path-aware attention, beam search)  (checked loops 7, 14)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 3 small files from `e77`, `e77_geo`, `e77_geo150`
- `figures/`: 2 images from the report sources
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
