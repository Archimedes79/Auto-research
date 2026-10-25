# E043_loop-3-system-grid-refiner-links

**Loop-3 system: grid + refiner + links (+ map prior, + temporal)**

| | |
|---|---|
| experiment id | E43 |
| theme | T2 - From the lane field to a graph (loops 1 - 4) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP3 graph assembly and junctions (links, connectors, assembler) |
| building blocks | BB01 convolution, U-Net / FPN pyramids |
| loop | loops02-03 |

## Result

no map F1 0.876, route F 0.744; split / merge F1 at 6 m only 0.34 / 0.38

## Techniques used (register K ids)

- K45: temporal map propagation and stream-query denoising, memory  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 10 small files from `e43`, `e43_joint`, `sys_e43_joint`
- `code/`: `make_e43.py`, `run_e43.sh`, `run_e43_joint.sh`, `run_e43c.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
