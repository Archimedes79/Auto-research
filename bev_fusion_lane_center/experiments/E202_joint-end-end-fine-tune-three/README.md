# E202_joint-end-end-fine-tune-three

**Joint end-to-end fine-tune, three arms (builder + route head + consumer)**

| | |
|---|---|
| experiment id | E202 |
| theme | T10 - Robustness, memory, map-conditioned builder, consumer under failure (loop 17) (loop 17) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop17 |

## Result

see the loop-17 report and experiment_execution/loop17_recovery (internal E136; e17/e136_*, e17_joint_*)

## Data

- `results/`: 20 small files from `e136_route`, `e17_joint_a`, `e17_joint_b`, `e17_joint_c`, `e17_joint_smoke`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
