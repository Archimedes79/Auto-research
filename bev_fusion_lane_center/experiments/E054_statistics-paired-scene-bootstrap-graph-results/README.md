# E054_statistics-paired-scene-bootstrap-graph-results

**Statistics: paired scene bootstrap of the graph results; seed noise of the dense head**

| | |
|---|---|
| experiment id | E54 / E55 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loop04 |

## Result

seed SD 0.0025 (VAL) / 0.0006 (TEST) F1; absolute F1 about +-0.035, paired differences about 0.005

## Data

- `code/`: `make_e54.py`, `make_e55.py`, `run_e54.sh`, `run_e55.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
