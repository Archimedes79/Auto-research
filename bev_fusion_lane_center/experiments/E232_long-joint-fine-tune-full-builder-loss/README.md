# E232_long-joint-fine-tune-full-builder-loss

**Longer joint fine-tune (4800 steps) of builder + route head with the builder loss at full weight, route loss through the builder (arm a) against a detached control (arm c)**

| | |
|---|---|
| experiment id | E232 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/history/indexes/LOOP18_STATE.md](../../experiment_execution/history/indexes/LOOP18_STATE.md) |

## Result

Failure-half route success a - c: night -0.036, blur -0.036, three cameras off 0.000 (rule +0.03 in two types not met), graph S level: the third negative for the route loss through the builder (E202, E214, E232); the failure-trained route head itself lifts route success +0.15 ... +0.37 over the base head, but its fine-tuned builder loses graph S with three cameras off (-0.029).

## Data

- `results/e232_table.json`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `code/` (all loop-18 scripts in their original layout: git tag `lcr/loop18-scratch`)
- raw outputs (arrays, checkpoints): `../../runs/`
