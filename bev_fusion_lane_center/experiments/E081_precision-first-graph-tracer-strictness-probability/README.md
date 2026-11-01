# E081_precision-first-graph-tracer-strictness-probability

**Precision-first graph: tracer strictness, probability filter, learned keep output**

| | |
|---|---|
| experiment id | E81 |
| theme | T4 - Perception-first graph with a learned assignment (reference system A) (loops 8 - 9) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP4 keep and verify (precision gates, verifiers, probabilities) |
| building blocks | BB05 peak picking: NMS, top-k, Gumbel top-k; BB13 relation bias, graph transformer, message passing |
| loop | loop08 - log: [../../experiment_execution/history/indexes/LOOP8_INDEX.md](../../experiment_execution/history/indexes/LOOP8_INDEX.md) |

## Result

keep 0.9 removes 95 % of the false lane length and keeps 53 % of the lane length

## Techniques used (register K ids)

- K48: nodes + pairwise edge classifier  (checked)
- K14: iterative graph growth guided by a CNN decision  (checked (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 2 small files from `e81_sparse`, `e81_sparse_ens`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
