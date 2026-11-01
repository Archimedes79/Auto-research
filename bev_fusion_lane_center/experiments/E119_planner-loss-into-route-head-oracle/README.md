# E119_planner-loss-into-route-head-oracle

**Planner loss into the route head; oracle route channel; route head from random weights**

| | |
|---|---|
| experiment id | E119 |
| theme | T7 - Planner interface, trust and the planner loss (loops 5 - 14) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP7 sparse graph as planner interface (tokens, consumer) |
| loop | loop14 - log: [../../experiment_execution/history/indexes/LOOP14_INDEX.md](../../experiment_execution/history/indexes/LOOP14_INDEX.md) |

## Result

no gain (0.34 m either way, oracle 0.316 m); planner loss alone does not create the attribute (AUROC 0.53 - 0.55)

## Techniques used (register K ids)

- K75: assignment of perceived lanes to SD roads (path-aware attention, beam search)  (checked loops 7, 14)
- K49: map uncertainty consumed by the planner; ego-status caveat of open-loop metrics  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 10 small files from `e119_a`, `e119_oracle`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
