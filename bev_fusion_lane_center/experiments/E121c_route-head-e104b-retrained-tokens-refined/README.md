# E121c_route-head-e104b-retrained-tokens-refined

**Route head E104b retrained on the tokens of the refined graph (same recipe, 40,000 iterations) and read on the refined graph; the unchanged head for comparison**

| | |
|---|---|
| experiment id | E121c |
| theme | T8 - Differentiable vertex positions (refinement of the traced graph) (loop 15) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB12 cross-attention |
| loop | loop15 - log: [../../experiment_execution/indexes/LOOP15_INDEX.md](../../experiment_execution/indexes/LOOP15_INDEX.md) |

## Result

route success 0.875 / 0.868 / 0.854 / 0.845 at map error 0 / 1 / 2 / 3 m against 0.853 / 0.843 / 0.832 / 0.820 on the traced graph (+0.022 to +0.026; the paired scene interval includes 0 at 0 m); the unchanged head reads the refined graph worse (-0.011 to -0.015); gap to system A 0.001 - 0.022; the planner consumer with these probabilities stays at 0.34 / 0.33 / 0.77 / 0.73 m (E121d); a second training run of each head gives 0.864 / 0.860 / 0.862 / 0.850 (traced) and 0.868 / 0.862 / 0.853 / 0.834 (refined): the route head varies by up to 0.03 between seeds, no effect of the refinement shown

## Techniques used (register K ids)

- K75: assignment of perceived lanes to SD roads (path-aware attention, beam search)  (checked loops 7, 14)
- K17: route head (cross-attention to SD-map tokens), planner consumer  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
