# E187_loop-16b-planner-consumer-validation

**Loop 16b (validation only, after the champion freeze): Planner consumer on validation (3 seeds, exact map)**

| | |
|---|---|
| experiment id | E187 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB11 self-attention; BB13 relation bias, graph transformer, message passing |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

path error at 20 m champion with the vertex head 0.31 m, with the head along the edges 0.26 m, control 0.27 m

## Techniques used (register K ids)

- K185: route attribute along the edges: the route head is asked at points every <= 4 m of every edge (linear interpolation of the two vertex tokens  (own idea, tested in loop 16b: positive)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
