# E086b_first-learned-consumer-graph-planner-loss

**First learned consumer of the graph; planner loss into the assignment network**

| | |
|---|---|
| experiment id | E86b / E86c |
| theme | T4 - Perception-first graph with a learned assignment (reference system A) (loops 8 - 9) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP7 sparse graph as planner interface (tokens, consumer) |
| building blocks | BB12 cross-attention |
| loop | loop08 - log: [../../experiment_execution/history/indexes/LOOP8_INDEX.md](../../experiment_execution/history/indexes/LOOP8_INDEX.md) |

## Result

consumer 0.88 to 0.41 m with the route attribute; planner loss no clear gain (route success 0.868 to 0.769)

## Techniques used (register K ids)

- K88: planner with map queries into the motion stage, multi-stage training  (checked loops 8, 15)
- K89: vectorised scene representation; ego query cross-attends map queries  (checked loops 8, 15)
- K90: sparse scene representation with anchor polylines  (checked loops 8, 15)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
