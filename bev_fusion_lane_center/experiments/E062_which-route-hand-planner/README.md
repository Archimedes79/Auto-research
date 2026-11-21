# E062_which-route-hand-planner

**Which route to hand to a planner (open-loop planning check)**

| | |
|---|---|
| experiment id | E62 |
| theme | T7 - Planner interface, trust and the planner loss (loops 5 - 14) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP7 sparse graph as planner interface (tokens, consumer) |
| loop | loop05 |

## Result

registered map route 0.35 m better than the perceived route; perceived route no better than straight ahead

## Techniques used (register K ids)

- K108: open-loop planning metrics and the ego-status caveat  (checked loop 8)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 15 small files from `e62_boot.json`, `e62_d03`, `e62_dbg`, `e62_s0`, `e62_s1_noreg`, `e62_s1_reg`, `e62_s2_reg`, `e62p_blur` ...
- code: removed in the clean-up of 2026-11-21, in the git history (commit `57b9eab1`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
