# E121a_planner-consumer-refined-graph-planner-loss

**Planner consumer (E117) on the refined graph and the planner loss through the live vertex positions: retrained consumer, plug, fine-tuned, joint with and without the position loss, keep probability as input**

| | |
|---|---|
| experiment id | E121a / E121b |
| theme | T8 - Differentiable vertex positions (refinement of the traced graph) (loop 15) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP7 sparse graph as planner interface (tokens, consumer) |
| building blocks | BB12 cross-attention |
| loop | loop15 - log: [../../experiment_execution/indexes/LOOP15_INDEX.md](../../experiment_execution/indexes/LOOP15_INDEX.md) |

## Result

path error at 20 m 0.36 m against 0.34 m (wrong / no map 0.76 / 0.75 against 0.76 / 0.74): no gain; the gradient reaches the positions; with the planner loss alone the refinement is undone (F1@1 0.852), with the position loss it is kept (0.863, planner 0.359 m)

## Techniques used (register K ids)

- K88: planner with map queries into the motion stage, multi-stage training  (checked loops 8, 15)
- K89: vectorised scene representation; ego query cross-attends map queries  (checked loops 8, 15)
- K49: map uncertainty consumed by the planner; ego-status caveat of open-loop metrics  (checked)
- K17: route head (cross-attention to SD-map tokens), planner consumer  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
