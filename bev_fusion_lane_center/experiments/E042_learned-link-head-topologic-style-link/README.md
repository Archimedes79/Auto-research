# E042_learned-link-head-topologic-style-link

**Learned link head; TopoLogic-style link scoring for splits and merges**

| | |
|---|---|
| experiment id | E42 / E45 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP3 graph assembly and junctions (links, connectors, assembler) |
| building blocks | BB13 relation bias, graph transformer, message passing |
| loop | loops02-03 |

## Result

within 0.01 of the rule links

## Techniques used (register K ids)

- K30: geometric link prior: endpoint distance / heading -> learned link probability  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- code: removed in the clean-up of 2026-11-21, in the git history (commit `57b9eab1`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
