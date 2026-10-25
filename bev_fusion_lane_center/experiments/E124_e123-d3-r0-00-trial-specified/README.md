# E124_e123-d3-r0-00-trial-specified

**E123.D3.r0.00 (T): trial as specified in the L8 row (second attempt after the verification-target flag fix; the first attempt is archived as e123_D3_r0_00_flagbug)**

| | |
|---|---|
| experiment id | E124 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB24 n-best hypotheses, beam, fixed-shape tree; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.697, F1@1 0.846, F1@0.5 0.657, J 0.107, 9.3 vertices per 100 m, false length 0.061, tau 0.6; VAL subset of 250 frames, 1 u; with beam 192 and 96 seeds S 0.696

## Techniques used (register K ids)

- K01: teardrop pattern: short rays in all directions, long narrow forward rays, samples on shared rings  (internal)
- K02: verification along a ray, survival / discrete-time hazard (same algebra as volume-rendering transmittance)  (NeRF from memory)
- K03: n best endpoints as hypotheses; next N vertices with validity in a local window  (checked)
- K04: deformable sampling at predicted offsets, reference points refined per layer  (checked)
- K06: sparse instances with fixed + learnable keypoints, recurrent instance propagation with ego compensation  (checked)
- K11: DAG of lane nodes with continue / fork / merge states  (checked)
- K14: iterative graph growth guided by a CNN decision  (checked (abstract))
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)
- K21: soft non-maximum suppression  (from memory)
- K23: scheduled sampling, DAgger (own states replace expert states)  (from memory)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 8 small files from `cards`, `e123_D3_r0_00`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
