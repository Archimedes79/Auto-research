# E186_loop-16b-point-spacing-route-attribute

**Loop 16b (validation only, after the champion freeze): Point spacing of the route attribute, champion, seed 0, validation**

| | |
|---|---|
| experiment id | E186 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB11 self-attention |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

8 m (98 points per frame) 0.872, 4 m (162) 0.888, 2 m (284) 0.899, vertex-based (68) 0.798

## Techniques used (register K ids)

- K185: route attribute along the edges: the route head is asked at points every <= 4 m of every edge (linear interpolation of the two vertex tokens  (own idea, tested in loop 16b: positive)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- no separate data files: the numbers above come from the catalogue; see the loop log
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
