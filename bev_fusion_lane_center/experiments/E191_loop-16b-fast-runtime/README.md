# E191_loop-16b-fast-runtime

**Loop 16b (validation only, after the champion freeze): Fast runtime**

| | |
|---|---|
| experiment id | E191 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB26 tree -> DAG merge and edge split (geometric) |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

builder in a CUDA graph (82 -> 19 ms under load; idle 69 -> 12 ms with the time recursion, 6 ms single frame), assembly vectorised 50 -> 20 ms under load (15 ms idle) with identical graphs, route head in padded CUDA graphs 6 -> 2 ms; chain 33 ms per frame (f1), 29 ms (j1) against 594 ms for the control

## Techniques used (register K ids)

- K186: CUDA-graph capture of the builder forward (and of the time-recursion prior), one host transfer per frame, padded CUDA graphs of the route he  (own application, tested in loop 16b: positive)
- K187: vectorised union and anchoring in the assembly (growing vertex array, sorted edge list, shared maximum filters, vectorised fork relocation)   (own, tested in loop 16b: identical graphs in 120 / 120 frames)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
