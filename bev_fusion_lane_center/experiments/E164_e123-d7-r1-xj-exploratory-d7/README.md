# E164_e123-d7-r1-xj-exploratory-d7

**E123.D7.r1.xj (X): exploratory (not a ladder row): D7 with the junction head (split / merge heat maps against the ground-truth landmarks), beam 192, seeds 96, continuation 2 u from the trial**

| | |
|---|---|
| experiment id | E164 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.717, F1@1 0.861, F1@0.5 0.680, J 0.140, 11.8 vertices per 100 m, false length 0.033, tau 0.8; VAL subset of 250 frames, 2 u

## Techniques used (register K ids)

- K31: explicit endpoint detection, point-lane attention  (checked)
- K33: fine-grained keypoint priors, topological denoising  (checked)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K180: junction anchoring: a heat-map peak of the junction head forks (merges) the nearest edge and joins a free branch start within 8 m  (own idea, tested in loop 16 (rung 3))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D7_r1_xj`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
