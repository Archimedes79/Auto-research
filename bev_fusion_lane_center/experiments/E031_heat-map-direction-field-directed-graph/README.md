# E031_heat-map-direction-field-directed-graph

**Heat map + direction field to a directed graph with the tracer (step size, point spacing)**

| | |
|---|---|
| experiment id | E31 / E33 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP2 line forming (dense field to lines: threshold, thinning, tracer) |
| building blocks | BB01 convolution, U-Net / FPN pyramids |
| loop | loops02-03 |

## Result

F1 0.879 / 0.851 (VAL / TEST), route F 0.765 / 0.702; even a perfect field gives route F 0.838

## Data

- `code/`: `make_e31_e33.py` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
