# E122_control-arm-r-trace-network-e94

**Control arm R: the trace network E94 trained from the shared E94 stem at 1 / 2 / 4 / 8 units (1 u = 600 iterations of 16 frames), seeds 0 and 1, fixed VAL subset of 250 frames; the full E94 runs on the same subset**

| | |
|---|---|
| experiment id | E122 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB11 self-attention; BB19 output heads: class + residual, splines; BB21 discrete choices inside the network; BB14 pointer attention |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S (composite of the plan) 0.713, 0.710, 0.707, 0.725, 0.724 (seed 0), seed 1 0.692, 0.712, 0.713, 0.726, 0.724; the shared stem carries the skill, the curve is flat from 2 u; E94 F1@1 0.883 on this subset, 21.8 vertices per 100 m, J 0.39

## Techniques used (register K ids)

- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)
- K15: matching-free point-to-polyline refinement of a traced graph  (internal)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 32 small files from `e122_s0_1u`, `e122_s0_2u`, `e122_s0_4u`, `e122_s0_8u`, `e122_s1_1u`, `e122_s1_2u`, `e122_s1_4u`, `e122_s1_8u`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
