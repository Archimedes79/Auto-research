# E019_line-forming-tuned-threshold-smoothing-minimum

**Line forming: tuned threshold, smoothing, minimum length; point-by-point line tracer against skeleton**

| | |
|---|---|
| experiment id | E19 / E26 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP2 line forming (dense field to lines: threshold, thinning, tracer) |
| loop | loop01 |

## Result

0.613 to 0.670 by threshold alone; tracer +0.023 (0.693); a perfect field gives 0.997

## Data

- `code/`: `e26_text.py`, `make_report_e26.py`, `plot_e26.py` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
