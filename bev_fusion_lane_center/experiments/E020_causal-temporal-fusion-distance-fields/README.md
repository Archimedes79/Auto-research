# E020_causal-temporal-fusion-distance-fields

**Causal temporal fusion of the distance fields (CRN)**

| | |
|---|---|
| experiment id | E20 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB07 recurrent cells (GRU, ConvGRU) |
| loop | loop01 |

## Result

0.675 against 0.670

## Techniques used (register K ids)

- K64: temporal self-attention over ego-aligned BEV features  (from memory)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
