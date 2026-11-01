# E152_e123-d6-r1-b2-l4-branch

**E123.D6.r1.b2 (F): L4 branch b2: SPF1.1 (survival only), SPF2.2 (hard-concrete keep gate), SPF3.2 (reach 12 m); shared base beam 192, seeds 96; continuation 2 u from e123_D6_r0_01**

| | |
|---|---|
| experiment id | E152 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.708, F1@1 0.839, F1@0.5 0.671, J 0.142, 14.3 vertices per 100 m, false length 0.039, tau 0.1; VAL subset of 250 frames, 2 u

## Techniques used (register K ids)

- K01: teardrop pattern: short rays in all directions, long narrow forward rays, samples on shared rings  (internal)
- K04: deformable sampling at predicted offsets, reference points refined per layer  (checked)
- K06: sparse instances with fixed + learnable keypoints, recurrent instance propagation with ego compensation  (checked)
- K08: expectation over a softmax (soft-argmax, integral regression) and its bias  (checked)
- K09: learned duplicate removal as a relation module  (checked)
- K10: lane segments + topology head  (checked)
- K12: path-level supervision instead of pieces  (checked)
- K129: learned sparsity with hard-concrete gates and an expected L0 penalty  (link checked 2026-10-23)
- K13: vertex and edge maps, GNN adjacency; local candidate edges  (checked / internal)
- K134: hard-concrete masks plus a Lagrangian pull to a target sparsity (prescribed density)  (link checked 2026-10-23)
- K135: learned edge keep gate with an expected-kept-edges penalty  (link checked 2026-10-23)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)
- K23: scheduled sampling, DAgger (own states replace expert states)  (from memory)
- K25: Douglas-Peucker simplification (oracle vertex density)  (from memory)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D6_r1_b2`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
