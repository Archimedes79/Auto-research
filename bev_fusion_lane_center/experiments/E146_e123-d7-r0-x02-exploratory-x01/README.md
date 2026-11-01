# E146_e123-d7-r0-x02-exploratory-x01

**E123.D7.r0.x02 (X): exploratory (not a ladder row): as x01 plus the best-per-ground-truth duplicate label (dupl = 1)**

| | |
|---|---|
| experiment id | E146 |
| theme | T9 - Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b) (loop 16) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP4 keep and verify (precision gates, verifiers, probabilities); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB01 convolution, U-Net / FPN pyramids; BB03 grid sampling, spatial transformer, warping; BB13 relation bias, graph transformer, message passing; BB14 pointer attention; BB22 survival / transmittance verification along a ray; BB23 teardrop (distance-dependent polar) sampling pattern; BB26 tree -> DAG merge and edge split (geometric); BB29 matching-free point-to-polyline loss; BB30 labels computed from geometry |
| loop | loop16 - log: [../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md](../../experiment_execution/history/indexes/TRIAGE_STATE_loop16.md) |

## Result

S 0.718, F1@1 0.864, F1@0.5 0.674, J 0.149, 12.2 vertices per 100 m, false length 0.040, tau 0.7; VAL subset of 250 frames, 1 u

## Techniques used (register K ids)

- K38: multiple-choice learning with score head and annealing  (checked)
- K39: entmax / sparsemax; soft top-k by optimal transport  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 4 small files from `cards`, `e123_D7_r0_x02`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
