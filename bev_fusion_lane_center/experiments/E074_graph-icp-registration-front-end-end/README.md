# E074_graph-icp-registration-front-end-end

**Graph-ICP registration in front of the end-to-end head**

| | |
|---|---|
| experiment id | E74 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB04 correlation / cost volume + soft-argmax |
| loop | loop06 |

## Result

route success nearly flat in the map error (+0.06 over the geometric transfer at 3 m)

## Techniques used (register K ids)

- K99: differentiable localisation of a BEV raster against a navigation map (pose volume, soft-argmax)  (checked loop 8)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `figures/`: 1 images from the report sources
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
