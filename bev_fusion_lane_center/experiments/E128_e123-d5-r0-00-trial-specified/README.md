# E128_e123-d5-r0-00-trial-specified

**E123.D5.r0.00 (T): trial as specified in the L8 row**

| | |
|---|---|
| experiment id | E128 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.673, F1@1 0.819, F1@0.5 0.652, J 0.104, 16.3 vertices per 100 m, false length 0.039, tau 0.7; VAL subset of 250 frames, 1 u; with beam 192 and 96 seeds S 0.680

## Techniques used (register K ids)

- K04: deformable sampling at predicted offsets, reference points refined per layer  (checked)
- K06: sparse instances with fixed + learnable keypoints, recurrent instance propagation with ego compensation  (checked)
- K08: expectation over a softmax (soft-argmax, integral regression) and its bias  (checked)
- K10: lane segments + topology head  (checked)
- K12: path-level supervision instead of pieces  (checked)
- K13: vertex and edge maps, GNN adjacency; local candidate edges  (checked / internal)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)
- K21: soft non-maximum suppression  (from memory)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 6 small files from `cards`, `e123_D5_r0_00`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
