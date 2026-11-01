# E184_loop-16b-edge-level-diagnosis-route

**Loop 16b (validation only, after the champion freeze): Edge-level diagnosis of the route flagging (p16_route_diag2.py)**

| | |
|---|---|
| experiment id | E184 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB11 self-attention |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

the same miss rate of route edges in both graphs (13.0 % / 13.5 %), confident misses, edges next to a fork vertex missed 28.7 % (champion) against 18.4 %; mean missed edge 8.1 m against 4.6 m; coverage ceiling 0.972 -> 0.872 against 0.993 -> 0.936

## Techniques used (register K ids)

- K185: route attribute along the edges: the route head is asked at points every <= 4 m of every edge (linear interpolation of the two vertex tokens  (own idea, tested in loop 16b: positive)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
