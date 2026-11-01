# E174_e123-d7-r3-j1-rung-3

**E123.D7.r3.j1 (F): rung 3 / junction head: split and merge heat maps (stem features, Gaussian sigma 2.5 m against the ground-truth landmarks) trained beside the builder, anchoring of forks and merges at the heat peaks in the assembly; continuation 8 u from the rung-2 winner**

| | |
|---|---|
| experiment id | E174 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.720, F1@1 0.866, F1@0.5 0.694, J 0.186, 15.3 vertices per 100 m, false length 0.037, tau 0.2; VAL subset of 250 frames, 8 u

## Techniques used (register K ids)

- K31: explicit endpoint detection, point-lane attention  (checked)
- K33: fine-grained keypoint priors, topological denoising  (checked)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K180: junction anchoring: a heat-map peak of the junction head forks (merges) the nearest edge and joins a free branch start within 8 m  (own idea, tested in loop 16 (rung 3))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 5 small files from `cards`, `e123_D7_r3_j1`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
