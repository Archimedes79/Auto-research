# E183_loop-16b-route-head-token-content

**Loop 16b (validation only, after the champion freeze): Route head with the token content set to zero (positions, cross sections, SD map stay)**

| | |
|---|---|
| experiment id | E183 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB11 self-attention; BB13 relation bias, graph transformer, message passing |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

champion 0.803 / 0.797 / 0.793 / 0.784, control 0.904 / 0.891 / 0.885 / 0.870 (map error 0 / 1 / 2 / 3 m); the head does not read the node features

## Techniques used (register K ids)

- K185: route attribute along the edges: the route head is asked at points every <= 4 m of every edge (linear interpolation of the two vertex tokens  (own idea, tested in loop 16b: positive)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
