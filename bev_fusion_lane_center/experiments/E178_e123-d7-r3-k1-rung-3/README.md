# E178_e123-d7-r3-k1-rung-3

**E123.D7.r3.k1 (F): rung 3 / fork coverage (FC, new loss term): a tip up to 14 m before a ground-truth fork (merge for backward tips) must cover the ground-truth lane points 10 - 16 m downstream with its children (min over the children, slack 1.2 m, smooth-L1; weight 0.5); junction-head run j1 continued 4 u at beam 288 / 144 (control g0: the same without the term)**

| | |
|---|---|
| experiment id | E178 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.725, F1@1 0.878, F1@0.5 0.701, J 0.178, 15.2 vertices per 100 m, false length 0.037, tau 0.5; VAL subset of 250 frames, 4 u

## Techniques used (register K ids)

- K03: n best endpoints as hypotheses; next N vertices with validity in a local window  (checked)
- K38: multiple-choice learning with score head and annealing  (checked)
- K31: explicit endpoint detection, point-lane attention  (checked)
- K182: fork-coverage loss: the children of a tip before a true fork must cover the ground-truth points 10 - 16 m downstream (set coverage, min over  (own idea, tested in loop 16: negative)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D7_r3_k1`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
