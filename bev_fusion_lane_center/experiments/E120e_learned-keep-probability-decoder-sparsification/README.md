# E120e_learned-keep-probability-decoder-sparsification

**Learned keep probability of the decoder as sparsification (hard gate on the vertices to measure it)**

| | |
|---|---|
| experiment id | E120e |
| theme | T8 - Differentiable vertex positions (refinement of the traced graph) (loop 15) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP4 keep and verify (precision gates, verifiers, probabilities); SP6 vertex positions of the sparse graph (refinement, keep gate) |
| building blocks | BB27 learned sparsity gates and soft selection |
| loop | loop15 - log: [../../experiment_execution/history/indexes/LOOP15_INDEX.md](../../experiment_execution/history/indexes/LOOP15_INDEX.md) |

## Result

threshold 0.5: F1@1 0.872 (precision 0.845 to 0.877, recall 0.882 to 0.866), false length 0.082 to 0.053; AUROC 0.868

## Techniques used (register K ids)

- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K09: learned duplicate removal as a relation module  (checked)
- K129: learned sparsity with hard-concrete gates and an expected L0 penalty  (link checked 2026-10-23)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
