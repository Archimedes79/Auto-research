# E120d_same-decoder-displacement-across-lane-only

**The same decoder with displacement across the lane only**

| | |
|---|---|
| experiment id | E120d |
| theme | T8 - Differentiable vertex positions (refinement of the traced graph) (loop 15) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP6 vertex positions of the sparse graph (refinement, keep gate) |
| building blocks | BB29 matching-free point-to-polyline loss |
| loop | loop15 - log: [../../experiment_execution/history/indexes/LOOP15_INDEX.md](../../experiment_execution/history/indexes/LOOP15_INDEX.md) |

## Result

TEST F1@0.5 0.692, F1@1 0.860 (sliding variants 0.863), vertex error 0.49 m instead of 0.43 m: the F1 gain is a lateral correction, half of the vertex-error gain is sliding along the lane

## Techniques used (register K ids)

- K15: matching-free point-to-polyline refinement of a traced graph  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
