# E170_e123-d6-r2-b2-l4-branch

**E123.D6.r2.b2 (F): L4 branch b2: HF1.1 (stage B off), HF2.2 (stage C on: attention over parent and siblings), HF3.2 (b = 3 children with multiple-choice learning); continuation 4 u from e123_D6_r1_b2**

| | |
|---|---|
| experiment id | E170 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.669, F1@1 0.792, F1@0.5 0.623, J 0.084, 14.7 vertices per 100 m, false length 0.037, tau 0.1; VAL subset of 250 frames, 4 u

## Techniques used (register K ids)

- K03: n best endpoints as hypotheses; next N vertices with validity in a local window  (checked)
- K04: deformable sampling at predicted offsets, reference points refined per layer  (checked)
- K06: sparse instances with fixed + learnable keypoints, recurrent instance propagation with ego compensation  (checked)
- K08: expectation over a softmax (soft-argmax, integral regression) and its bias  (checked)
- K09: learned duplicate removal as a relation module  (checked)
- K10: lane segments + topology head  (checked)
- K12: path-level supervision instead of pieces  (checked)
- K13: vertex and edge maps, GNN adjacency; local candidate edges  (checked / internal)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)
- K22: cascaded refinement with stage-wise losses  (from memory)
- K23: scheduled sampling, DAgger (own states replace expert states)  (from memory)
- K34: joint predecessor / successor modelling  (checked)
- K38: multiple-choice learning with score head and annealing  (checked)
- K39: entmax / sparsemax; soft top-k by optimal transport  (checked)
- K43: look forward twice, contrastive denoising  (checked)
- K44: quality estimation head for calibrated confidence  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D6_r2_b2`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
