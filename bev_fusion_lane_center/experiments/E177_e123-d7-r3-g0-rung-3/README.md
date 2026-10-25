# E177_e123-d7-r3-g0-rung-3

**E123.D7.r3.g0 (F): rung 3 / capacity: the junction-head run j1 trained 4 u more at beam M = 288 and 144 seeds (j1: 192 / 96)**

| | |
|---|---|
| experiment id | E177 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.721, F1@1 0.878, F1@0.5 0.704, J 0.155, 15.2 vertices per 100 m, false length 0.035, tau 0.7; VAL subset of 250 frames, 4 u

## Techniques used (register K ids)

- K03: n best endpoints as hypotheses; next N vertices with validity in a local window  (checked)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D7_r3_g0`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
