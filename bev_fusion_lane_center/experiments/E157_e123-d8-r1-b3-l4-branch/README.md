# E157_e123-d8-r1-b3-l4-branch

**E123.D8.r1.b3 (F): L4 branch b3: SPF1.2 (vertex-count penalty with weight 1), SPF2.1 (validity = survival), SPF3.2 (reach 12 m); shared base beam 192, seeds 96; continuation 2 u from e123_D8_r0_01**

| | |
|---|---|
| experiment id | E157 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.690, F1@1 0.841, F1@0.5 0.644, J 0.088, 12.1 vertices per 100 m, false length 0.059, tau 0.9; VAL subset of 250 frames, 2 u

## Techniques used (register K ids)

- K01: teardrop pattern: short rays in all directions, long narrow forward rays, samples on shared rings  (internal)
- K02: verification along a ray, survival / discrete-time hazard (same algebra as volume-rendering transmittance)  (NeRF from memory)
- K03: n best endpoints as hypotheses; next N vertices with validity in a local window  (checked)
- K04: deformable sampling at predicted offsets, reference points refined per layer  (checked)
- K09: learned duplicate removal as a relation module  (checked)
- K10: lane segments + topology head  (checked)
- K12: path-level supervision instead of pieces  (checked)
- K129: learned sparsity with hard-concrete gates and an expected L0 penalty  (link checked 2026-10-23)
- K13: vertex and edge maps, GNN adjacency; local candidate edges  (checked / internal)
- K134: hard-concrete masks plus a Lagrangian pull to a target sparsity (prescribed density)  (link checked 2026-10-23)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K25: Douglas-Peucker simplification (oracle vertex density)  (from memory)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D8_r1_b3`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
