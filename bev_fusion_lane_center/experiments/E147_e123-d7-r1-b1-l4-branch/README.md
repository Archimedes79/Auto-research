# E147_e123-d7-r1-b1-l4-branch

**E123.D7.r1.b1 (F): L4 branch b1: SPF1.1 (survival only), SPF2.1 (validity = survival), SPF3.1 (reach 24 m); shared base beam 192, seeds 96; continuation 2 u from e123_D7_r0_00**

| | |
|---|---|
| experiment id | E147 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.718, F1@1 0.862, F1@0.5 0.679, J 0.149, 11.8 vertices per 100 m, false length 0.043, tau 0.7; VAL subset of 250 frames, 2 u

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

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D7_r1_b1`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
