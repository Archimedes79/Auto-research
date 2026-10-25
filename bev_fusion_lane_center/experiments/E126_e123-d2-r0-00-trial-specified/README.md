# E126_e123-d2-r0-00-trial-specified

**E123.D2.r0.00 (T): trial as specified in the L8 row**

| | |
|---|---|
| experiment id | E126 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB08 iterative refinement, cascades; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB24 n-best hypotheses, beam, fixed-shape tree; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/indexes/TRIAGE_STATE.md](../../experiment_execution/indexes/TRIAGE_STATE.md) |

## Result

S 0.623, F1@1 0.776, F1@0.5 0.608, J 0.077, 19.5 vertices per 100 m, false length 0.048, tau 0.8; VAL subset of 250 frames, 1 u

## Techniques used (register K ids)

- K04: deformable sampling at predicted offsets, reference points refined per layer  (checked)
- K08: expectation over a softmax (soft-argmax, integral regression) and its bias  (checked)
- K09: learned duplicate removal as a relation module  (checked)
- K11: DAG of lane nodes with continue / fork / merge states  (checked)
- K14: iterative graph growth guided by a CNN decision  (checked (abstract))
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)
- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)
- K23: scheduled sampling, DAgger (own states replace expert states)  (from memory)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D2_r0_00`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
