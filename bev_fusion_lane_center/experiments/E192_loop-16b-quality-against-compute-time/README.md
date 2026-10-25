# E192_loop-16b-quality-against-compute-time

**Loop 16b (validation only, after the champion freeze): Quality against compute time (VAL250, idle GPU)**

| | |
|---|---|
| experiment id | E192 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB28 log-odds temporal fusion; BB24 n-best hypotheses, beam, fixed-shape tree |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

j1 192 / 96 S 0.747 in 20 ms, j1 288 / 144 S 0.748 in 21 ms, f1 288 / 144 S 0.749 in 27 ms; below 192 / 96 recall collapses (96 / 48: S 0.70); the time recursion is a tie in S

## Techniques used (register K ids)

- K186: CUDA-graph capture of the builder forward (and of the time-recursion prior), one host transfer per frame, padded CUDA graphs of the route he  (own application, tested in loop 16b: positive)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
