# E083_assignment-network-6-m-edges-rule

**Assignment network on the 6 m edges of a rule-built graph: relation-biased transformer (0.27 M) with route and keep outputs**

| | |
|---|---|
| experiment id | E83 |
| theme | T4 - Perception-first graph with a learned assignment (reference system A) (loops 8 - 9) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP4 keep and verify (precision gates, verifiers, probabilities) |
| building blocks | BB13 relation bias, graph transformer, message passing; BB11 self-attention |
| loop | loop08 - log: [../../experiment_execution/indexes/LOOP8_INDEX.md](../../experiment_execution/indexes/LOOP8_INDEX.md) |

## Result

route success 0.881 / 0.879 / 0.868 / 0.839: +0.15 to +0.30 over the pixel-level head

## Techniques used (register K ids)

- K48: nodes + pairwise edge classifier  (checked)
- K75: assignment of perceived lanes to SD roads (path-aware attention, beam search)  (checked loops 7, 14)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 35 small files from `e83_a`, `e83_alt`, `e83_attr`, `e83_cleanonly`, `e83_e8`, `e83_e8_d128`, `e83_e8_d64`, `e83_e8_drop` ...
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
