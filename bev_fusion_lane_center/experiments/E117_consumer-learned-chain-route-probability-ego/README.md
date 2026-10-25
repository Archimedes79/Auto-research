# E117_consumer-learned-chain-route-probability-ego

**Consumer (93 k) on the learned chain: route probability, ego motion, soft / hard verifier inputs, step token**

| | |
|---|---|
| experiment id | E117 |
| theme | T7 - Planner interface, trust and the planner loss (loops 5 - 14) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP7 sparse graph as planner interface (tokens, consumer) |
| building blocks | BB12 cross-attention; BB11 self-attention |
| loop | loop14 - log: [../../experiment_execution/indexes/LOOP14_INDEX.md](../../experiment_execution/indexes/LOOP14_INDEX.md) |

## Result

0.90 to 0.40 m with the route probability, 0.34 m with ego motion (system A: 0.36 m); step token 0.56 m

## Techniques used (register K ids)

- K75: assignment of perceived lanes to SD roads (path-aware attention, beam search)  (checked loops 7, 14)
- K49: map uncertainty consumed by the planner; ego-status caveat of open-loop metrics  (checked)
- K95: map features gated by ego state, trained jointly with the planner  (checked loop 14 (abstract))
- K17: route head (cross-attention to SD-map tokens), planner consumer  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 15 small files from `e117_a`, `e117_b`, `e117_c`, `e117_d`, `e117_d2`, `e117_d2.log`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
