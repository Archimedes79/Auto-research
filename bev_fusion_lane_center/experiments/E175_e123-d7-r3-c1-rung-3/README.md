# E175_e123-d7-r3-c1-rung-3

**E123.D7.r3.c1 (N): rung 3 / length control: plain continuation 8 u from the rung-2 winner (no change set), to separate the effect of the flags from the effect of the extra training**

| | |
|---|---|
| experiment id | E175 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.724, F1@1 0.869, F1@0.5 0.692, J 0.189, 15.0 vertices per 100 m, false length 0.037, tau 0.3; VAL subset of 250 frames, 8 u

## Techniques used (register K ids)

- K15: matching-free point-to-polyline refinement of a traced graph  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D7_r3_c1`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
