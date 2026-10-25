# E067_held-registration-carried-forward-graph-map

**Held registration, carried-forward graph, map lanes as lane graph, jumps and failure bursts**

| | |
|---|---|
| experiment id | E67 - E72 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loop05 |

## Result

night: hold 0.53 m better than per frame; carry-forward 0.651 to 0.741 for about 5 s; map lanes 0.887

## Data

- `code/`: `boot_e71.py`, `make_e67.py`, `make_e68.py`, `make_e69.py`, `make_e70.py`, `make_e71.py`, `make_e72.py`, `run_e67.sh`, `run_e67seeds.sh`, `run_e68.sh`, `run_e69.sh`, `run_e70.sh`, `run_e71.sh`, `run_e72.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
