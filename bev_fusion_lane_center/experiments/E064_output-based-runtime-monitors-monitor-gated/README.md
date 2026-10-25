# E064_output-based-runtime-monitors-monitor-gated

**Output-based runtime monitors and a monitor-gated fallback to the map route**

| | |
|---|---|
| experiment id | E64 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loop05 |

## Result

failing frames found with AUROC 0.953; lateral error 1.03 to 0.43 m

## Data

- `results/`: 3 small files from `e64_monitors`
- `code/`: `make_e64.py`, `run_e64.sh`, `run_e64b.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
