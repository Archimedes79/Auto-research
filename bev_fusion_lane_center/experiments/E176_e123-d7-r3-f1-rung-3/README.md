# E176_e123-d7-r3-f1-rung-3

**E123.D7.r3.f1 (F): rung 3 / final-model candidate: junction head (parent j1) plus time recursion T.2 (previous-frame nodes warped by the ego motion as seed candidates, log-odds fusion of the validity); continuation 8 u**

| | |
|---|---|
| experiment id | E176 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.723, F1@1 0.864, F1@0.5 0.690, J 0.191, 15.1 vertices per 100 m, false length 0.036, tau 0.3; VAL subset of 250 frames, 8 u

## Techniques used (register K ids)

- K06: sparse instances with fixed + learnable keypoints, recurrent instance propagation with ego compensation  (checked)
- K24: log-odds fusion of evidence over time  (from memory)
- K31: explicit endpoint detection, point-lane attention  (checked)
- K33: fine-grained keypoint priors, topological denoising  (checked)
- K180: junction anchoring: a heat-map peak of the junction head forks (merges) the nearest edge and joins a free branch start within 8 m  (own idea, tested in loop 16 (rung 3))
- K181: fork relocation: a fork vertex moves up its stem to the foot point of a strong junction peak if the straight branch edges stay within a tole  (own idea, tested in loop 16 (rung 3))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 8 small files from `cards`, `e123_D7_r3_f1`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
