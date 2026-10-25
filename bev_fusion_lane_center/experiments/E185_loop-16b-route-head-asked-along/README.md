# E185_loop-16b-route-head-asked-along

**Loop 16b (validation only, after the champion freeze): Route head asked along the edges (points every <= 4 m, interpolated tokens), two seeds, validation**

| | |
|---|---|
| experiment id | E185 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB11 self-attention; BB13 relation bias, graph transformer, message passing; BB10 memory bank / query propagation |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

champion 0.888 / 0.889 (exact), 0.881 / 0.886 (1 m), 0.886 / 0.883 (2 m), 0.883 / 0.876 (3 m); control with the same treatment 0.876 / 0.894; vertex-based 0.798 and 0.881 / 0.875. The route gap of the sparse graph is closed

## Techniques used (register K ids)

- K185: route attribute along the edges: the route head is asked at points every <= 4 m of every edge (linear interpolation of the two vertex tokens  (own idea, tested in loop 16b: positive)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
