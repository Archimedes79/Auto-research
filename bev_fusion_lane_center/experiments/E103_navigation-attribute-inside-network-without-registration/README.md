# E103_navigation-attribute-inside-network-without-registration

**Navigation attribute inside the network without registration: step-token head, graph transformer, learned localisation**

| | |
|---|---|
| experiment id | E103 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB04 correlation / cost volume + soft-argmax; BB13 relation bias, graph transformer, message passing |
| loop | loop10 |

## Result

0.78 - 0.71 against 0.86 - 0.84 (rule graph + registration); 40,000 instead of 3,000 steps: +0.14 to +0.39

## Techniques used (register K ids)

- K99: differentiable localisation of a BEV raster against a navigation map (pose volume, soft-argmax)  (checked loop 8)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 51 small files from `e103_a`, `e103_smoke`, `e103b`, `e103c`, `e103c2`, `e103c3`, `e103d`, `e103d_smoke` ...
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
