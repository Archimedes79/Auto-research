# E066_small-learned-planner-module-output-does

**Small learned planner (0.3 M CNN) on the module output: does it use and follow the route?**

| | |
|---|---|
| experiment id | E66 |
| theme | T7 - Planner interface, trust and the planner loss (loops 5 - 14) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP7 sparse graph as planner interface (tokens, consumer) |
| building blocks | BB11 self-attention |
| loop | loop05 |

## Result

route channel -20 % lateral error on curves; with a wrong route no more compliance than without

## Techniques used (register K ids)

- K88: planner with map queries into the motion stage, multi-stage training  (checked loops 8, 15)
- K89: vectorised scene representation; ego query cross-attends map queries  (checked loops 8, 15)
- K90: sparse scene representation with anchor polylines  (checked loops 8, 15)
- K108: open-loop planning metrics and the ego-status caveat  (checked loop 8)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 2 small files from `e66_planner`
- `code/`: `make_e66.py` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
