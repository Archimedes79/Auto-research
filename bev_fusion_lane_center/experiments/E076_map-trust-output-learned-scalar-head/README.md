# E076_map-trust-output-learned-scalar-head

**Map-trust output: learned scalar head against a label-free geometric agreement score**

| | |
|---|---|
| experiment id | E76 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB17 conditioning and gating |
| loop | loop06 |

## Result

agreement score AUROC 0.996 where the learned scalar reaches 0.82

## Techniques used (register K ids)

- K68: linear probing then full fine-tuning  (checked loop 6 (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
