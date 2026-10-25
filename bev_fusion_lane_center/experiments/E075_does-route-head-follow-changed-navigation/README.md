# E075_does-route-head-follow-changed-navigation

**Does the route head follow a changed navigation goal (counterfactual routes)?**

| | |
|---|---|
| experiment id | E75 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB17 conditioning and gating |
| loop | loop06 |

## Result

alternative route 0.828 against 0.887 for the driven one (geometric transfer 0.805 / 0.887; E60 0.535, E65d 0.699)

## Techniques used (register K ids)

- K93: route compliance metric and counterfactual commands  (checked loops 8, 14)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `figures/`: 1 images from the report sources
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
