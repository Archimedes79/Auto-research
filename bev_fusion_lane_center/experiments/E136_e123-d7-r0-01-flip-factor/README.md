# E136_e123-d7-r0-01-flip-factor

**E123.D7.r0.01 (F): flip factor G (revisit) to level 2: E120-style revisit of the whole node set (continuation, new revisit layers)**

| | |
|---|---|
| experiment id | E136 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.700, F1@1 0.833, F1@0.5 0.652, J 0.126, 11.8 vertices per 100 m, false length 0.038, tau 0.8; VAL subset of 250 frames, 1 u; with beam 192 and 96 seeds S 0.710

## Techniques used (register K ids)

- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K43: look forward twice, contrastive denoising  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 5 small files from `cards`, `e123_D7_r0_01`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
