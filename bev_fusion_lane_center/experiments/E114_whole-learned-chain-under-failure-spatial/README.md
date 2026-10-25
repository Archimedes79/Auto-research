# E114_whole-learned-chain-under-failure-spatial

**Whole learned chain under failure with the spatial hold and a route confidence**

| | |
|---|---|
| experiment id | E114 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB10 memory bank / query propagation |
| loop | loop13 - log: [../../experiment_execution/indexes/LOOP13_INDEX.md](../../experiment_execution/indexes/LOOP13_INDEX.md) |

## Result

route success night 0.09 to 0.44, blur 0.23 to 0.55

## Techniques used (register K ids)

- K79: propagate confident queries; fast and slow latent memory  (checked loops 7, 14)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 3 small files from `e114_a`, `e114_tok`
- `code/`: `_e114.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
