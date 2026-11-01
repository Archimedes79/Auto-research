# E118_consumer-under-sensor-failure-input-memory

**The consumer under sensor failure: input memory, failure-frame training, an unseen failure type**

| | |
|---|---|
| experiment id | E118 |
| theme | T7 - Planner interface, trust and the planner loss (loops 5 - 14) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP7 sparse graph as planner interface (tokens, consumer) |
| building blocks | BB10 memory bank / query propagation |
| loop | loop14 - log: [../../experiment_execution/history/indexes/LOOP14_INDEX.md](../../experiment_execution/history/indexes/LOOP14_INDEX.md) |

## Result

night 2.73 to 0.72 m (spatial hold) to 0.44 m (also failure training)

## Techniques used (register K ids)

- K82: maps that stay usable with missing views or sensors: reconstruction, distillation, masked-input training, gating  (checked loops 8, 14 (abstract))
- K79: propagate confident queries; fast and slow latent memory  (checked loops 7, 14)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 13 small files from `e118_a`, `e118_b`, `e118_c_blur`, `e118_c_cam_front3_off`, `e118_c_dark`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
