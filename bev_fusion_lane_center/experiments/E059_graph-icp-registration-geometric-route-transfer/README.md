# E059_graph-icp-registration-geometric-route-transfer

**Graph-ICP registration and geometric route transfer (no parameters), lane-level map**

| | |
|---|---|
| experiment id | E59 |
| theme | T3 - Map prior and navigation route (loops 1 - 7) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loop05 |

## Result

route success 0.923 / 0.883 / 0.782 at 1 / 2 / 3 m against 0.640 / 0.305 / 0.153 without registration

## Data

- `results/`: 56 small files from `e59_d0.1_any`, `e59_d0.1_keep`, `e59_d0.3_any`, `e59_d0.3_keep`, `e59_d0.5_any`, `e59_d0.5_keep`, `e59_nomap`, `e59_old` ...
- `code/`: `make_e59.py`, `run_e59.sh`, `run_e59_hmm.sh`, `run_e59b.sh`, `run_e59c.sh`, `run_e59d.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
