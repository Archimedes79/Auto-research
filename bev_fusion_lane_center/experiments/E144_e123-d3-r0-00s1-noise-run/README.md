# E144_e123-d3-r0-00s1-noise-run

**E123.D3.r0.00s1 (N): noise run: trial D3 with seed 1**

| | |
|---|---|
| experiment id | E144 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB24 n-best hypotheses, beam, fixed-shape tree; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.687, F1@1 0.830, F1@0.5 0.644, J 0.095, 9.3 vertices per 100 m, false length 0.063, tau 0.6; VAL subset of 250 frames, 1 u

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

- `results/`: 4 small files from `cards`, `e123_D3_r0_00s1`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
