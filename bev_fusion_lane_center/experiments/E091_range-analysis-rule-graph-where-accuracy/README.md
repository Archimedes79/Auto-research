# E091_range-analysis-rule-graph-where-accuracy

**Range analysis of the rule graph (TEST): where the accuracy is lost**

| | |
|---|---|
| experiment id | E91 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP5 learned sparse extractor (step choice, vertices, events) |
| loop | loop09 |

## Result

F1 0.957 (0 - 10 m) falls to 0.800 (40 - 50 m); a far lane is found for 92 % of the lines when the near part is found, 69 % when it is missed

## Techniques used (register K ids)

- K14: iterative graph growth guided by a CNN decision  (checked (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 2 small files from `e91_range`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
