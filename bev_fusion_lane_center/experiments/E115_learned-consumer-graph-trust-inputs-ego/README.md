# E115_learned-consumer-graph-trust-inputs-ego

**Learned consumer of the graph with trust inputs and ego motion**

| | |
|---|---|
| experiment id | E115 |
| theme | T7 - Planner interface, trust and the planner loss (loops 5 - 14) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP7 sparse graph as planner interface (tokens, consumer) |
| building blocks | BB17 conditioning and gating |
| loop | loop13 - log: [../../experiment_execution/indexes/LOOP13_INDEX.md](../../experiment_execution/indexes/LOOP13_INDEX.md) |

## Result

ego motion: pooled failure error 1.09 to 0.71 m; a failure flag adds nothing

## Techniques used (register K ids)

- K87: online map uncertainty consumed by a trajectory predictor  (checked loops 8, 12 (abstract))
- K95: map features gated by ego state, trained jointly with the planner  (checked loop 14 (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 13 small files from `e115_a`, `e115_b`, `e115_c`, `e115_d`, `e115_e`, `e115_f`, `e115_viz`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
