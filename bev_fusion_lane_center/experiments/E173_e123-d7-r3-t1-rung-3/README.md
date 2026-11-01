# E173_e123-d7-r3-t1-rung-3

**E123.D7.r3.t1 (F): rung 3 / time recursion T.2: previous-frame nodes warped by the ego motion as extra seed candidates and log-odds fusion of the validity (prior from the own single-frame prediction of the previous frame, pose noise and 10 % no-prior frames in training); continuation 8 u from the rung-2 winner**

| | |
|---|---|
| experiment id | E173 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.725, F1@1 0.863, F1@0.5 0.687, J 0.182, 14.5 vertices per 100 m, false length 0.034, tau 0.6; VAL subset of 250 frames, 8 u

## Techniques used (register K ids)

- K06: sparse instances with fixed + learnable keypoints, recurrent instance propagation with ego compensation  (checked)
- K24: log-odds fusion of evidence over time  (from memory)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D7_r3_t1`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
