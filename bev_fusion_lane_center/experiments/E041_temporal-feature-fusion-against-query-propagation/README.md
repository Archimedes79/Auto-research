# E041_temporal-feature-fusion-against-query-propagation

**Temporal feature fusion (warped previous frame) against query propagation**

| | |
|---|---|
| experiment id | E41 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB10 memory bank / query propagation; BB07 recurrent cells (GRU, ConvGRU) |
| loop | loops02-03 |

## Result

fusion +0.008 F1 and one third less flicker; query propagation 0

## Techniques used (register K ids)

- K45: temporal map propagation and stream-query denoising, memory  (checked)
- K103: lane graph from video: late graph merge against early BEV fusion  (checked loop 8 (html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 8 small files from `e41_A`, `e41_B`, `e41_Btr`, `sys_e41_A`, `sys_e41_Btr`
- `code/`: `make_e41.py`, `run_e41b.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
