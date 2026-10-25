# E168_e123-d7-r2-b4-l4-branch

**E123.D7.r2.b4 (F): L4 branch b4: HF1.2 (stage B on: small pattern around every child), HF2.2 (stage C on: attention over parent and siblings), HF3.1 (b = 2); continuation 4 u from e123_D7_r1_b1**

| | |
|---|---|
| experiment id | E168 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.709, F1@1 0.830, F1@0.5 0.670, J 0.169, 14.6 vertices per 100 m, false length 0.033, tau 0.5; VAL subset of 250 frames, 4 u

## Techniques used (register K ids)

- K01: teardrop pattern: short rays in all directions, long narrow forward rays, samples on shared rings  (internal)
- K02: verification along a ray, survival / discrete-time hazard (same algebra as volume-rendering transmittance)  (NeRF from memory)
- K03: n best endpoints as hypotheses; next N vertices with validity in a local window  (checked)
- K04: deformable sampling at predicted offsets, reference points refined per layer  (checked)
- K07: look-up at the current estimate, update, repeat  (checked)
- K10: lane segments + topology head  (checked)
- K12: path-level supervision instead of pieces  (checked)
- K13: vertex and edge maps, GNN adjacency; local candidate edges  (checked / internal)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K21: soft non-maximum suppression  (from memory)
- K22: cascaded refinement with stage-wise losses  (from memory)
- K23: scheduled sampling, DAgger (own states replace expert states)  (from memory)
- K34: joint predecessor / successor modelling  (checked)
- K36: lane / Bezier-aligned deformable attention  (checked)
- K43: look forward twice, contrastive denoising  (checked)
- K44: quality estimation head for calibrated confidence  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D7_r2_b4`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
