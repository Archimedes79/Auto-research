# E105_input-memory-front-trace-network-ego

**Input memory in front of the trace network: ego-motion warp, convex mix and a hold triggered by a label-free heat monitor**

| | |
|---|---|
| experiment id | E105 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB10 memory bank / query propagation; BB03 grid sampling, spatial transformer, warping |
| loop | loop11 - log: [../../experiment_execution/indexes/LOOP11_INDEX.md](../../experiment_execution/indexes/LOOP11_INDEX.md) |

## Result

night F1 0.337 to 0.728, blur 0.524 to 0.699

## Techniques used (register K ids)

- K64: temporal self-attention over ego-aligned BEV features  (from memory)
- K45: temporal map propagation and stream-query denoising, memory  (checked)
- K78: global raster memory accumulated from past predictions  (checked loop 7 (html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 5 small files from `e105_a`, `e105_monfwd`, `e105_monquad`, `e105_odo1`, `e105_odo3`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
