# E051_node-conditioned-dense-head-direction-consistency

**Node-conditioned dense head; direction consistency as acceptance criterion for connectors**

| | |
|---|---|
| experiment id | E51 / E56 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP3 graph assembly and junctions (links, connectors, assembler) |
| building blocks | BB17 conditioning and gating |
| loop | loop04 |

## Result

+0.003 F1; connectors lose 0.026 split F1

## Techniques used (register K ids)

- K48: nodes + pairwise edge classifier  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- code: removed in the clean-up of 2026-11-21, in the git history (commit `57b9eab1`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
