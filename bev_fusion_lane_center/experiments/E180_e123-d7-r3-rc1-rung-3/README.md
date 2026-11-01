# E180_e123-d7-r3-rc1-rung-3

**E123.D7.r3.rc1 (X): rung 3 / exploration after the test freeze (validation only, not a ladder row, not a candidate for the test): recall-weighted fine-tune: f1 with the soft-recall loss weight 3 instead of 1, 4 u, beam 288 / 144**

| | |
|---|---|
| experiment id | E180 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB10 memory bank / query propagation; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB25 cascade rescoring (generate, refine, rescore); BB26 tree -> DAG merge and edge split (geometric); BB27 learned sparsity gates and soft selection; BB28 log-odds temporal fusion; BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.722, F1@1 0.882, F1@0.5 0.699, J 0.132, 14.5 vertices per 100 m, false length 0.040, tau 0.8; VAL subset of 250 frames, 4 u

## Techniques used (register K ids)

- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K23: scheduled sampling, DAgger (own states replace expert states)  (from memory)
- K184: recall-weighted fine-tuning (soft-recall loss weight 3) of the champion  (own idea, tested in loop 16)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 5 small files from `cards`, `e123_D7_r3_rc1`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
