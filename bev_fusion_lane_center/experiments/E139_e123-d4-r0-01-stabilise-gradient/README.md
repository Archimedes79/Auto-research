# E139_e123-d4-r0-01-stabilise-gradient

**E123.D4.r0.01 (S): stabilise: gradient of the tree losses into the seed state x 0.1, lr 3e-4, continuation**

| | |
|---|---|
| experiment id | E139 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB24 n-best hypotheses, beam, fixed-shape tree; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.687, F1@1 0.834, F1@0.5 0.637, J 0.095, 10.1 vertices per 100 m, false length 0.061, tau 0.5; VAL subset of 250 frames, 1 u; with beam 192 and 96 seeds S 0.698

## Techniques used (register K ids)

- K23: scheduled sampling, DAgger (own states replace expert states)  (from memory)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 5 small files from `cards`, `e123_D4_r0_01`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
