# E181_e123-d7-r3-s2-rung-3

**E123.D7.r3.s2 (N): rung 3 / recipe replicate without the time recursion: teardrop, lattice, geometry dedup, global context, warm; stage C, b = 3 with MCL, junction head; beam 288 / 144; ONE run of 12 u from the shared stem, seed 1 (replicate s1 with the time recursion from step 0 failed: S 0.469, the seed head did not learn: the prior built from the network's own untrained single-frame prediction corrupts the seeds)**

| | |
|---|---|
| experiment id | E181 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.726, F1@1 0.870, F1@0.5 0.697, J 0.167, 14.6 vertices per 100 m, false length 0.032, tau 0.9; VAL subset of 250 frames, 12 u

## Techniques used (register K ids)

- K01: teardrop pattern: short rays in all directions, long narrow forward rays, samples on shared rings  (internal)
- K02: verification along a ray, survival / discrete-time hazard (same algebra as volume-rendering transmittance)  (NeRF from memory)
- K03: n best endpoints as hypotheses; next N vertices with validity in a local window  (checked)
- K04: deformable sampling at predicted offsets, reference points refined per layer  (checked)
- K10: lane segments + topology head  (checked)
- K12: path-level supervision instead of pieces  (checked)
- K13: vertex and edge maps, GNN adjacency; local candidate edges  (checked / internal)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K21: soft non-maximum suppression  (from memory)
- K23: scheduled sampling, DAgger (own states replace expert states)  (from memory)
- K31: explicit endpoint detection, point-lane attention  (checked)
- K33: fine-grained keypoint priors, topological denoising  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 5 small files from `cards`, `e123_D7_r3_s2`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
