# E140_e123-d1-r0-01-flip-factor

**E123.D1.r0.01 (F): flip factor B (pattern package) to level 2 = teardrop, restart from the shared stem, 2 u**

| | |
|---|---|
| experiment id | E140 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB24 n-best hypotheses, beam, fixed-shape tree; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.666, F1@1 0.801, F1@0.5 0.567, J 0.101, 13.7 vertices per 100 m, false length 0.050, tau 0.5; VAL subset of 250 frames, 2 u; with beam 192 and 96 seeds S 0.675

## Techniques used (register K ids)

- K01: teardrop pattern: short rays in all directions, long narrow forward rays, samples on shared rings  (internal)
- K02: verification along a ray, survival / discrete-time hazard (same algebra as volume-rendering transmittance)  (NeRF from memory)
- K03: n best endpoints as hypotheses; next N vertices with validity in a local window  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 5 small files from `cards`, `e123_D1_r0_01`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
