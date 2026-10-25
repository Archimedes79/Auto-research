# E213_rough-localisation-and-map-trust

**Rough localisation of the SD map (pose-grid soft-argmax + causal world-frame accumulation) and a map trust gate in front of the map-conditioned builder**

| | |
|---|---|
| experiment id | E213 |
| theme | T11 - Fork pruning, map localisation and failure-trained navigation (loop 18) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| loop | loop18 - state: [../../experiment_execution/indexes/LOOP18_STATE.md](../../experiment_execution/indexes/LOOP18_STATE.md) |

## Result

Pose error of the map 2.55 m -> 0.82 m (night failure half, sigma 2 m); failure-half S of builder + gate: none 0.651, unregistered 2 m 0.643, registered 0.664, exact 0.676 (night); paired F1@1 registered - none +0.035 [+0.018, +0.059]; wrong map 0.605 -> 0.647 with the trust gate 0.75 (none 0.651); exact map keeps 72 - 87 % of its benefit at 0.75, 88 - 99 % at 0.62 (best chain). S gate (none + 0.015) missed by a hair, F1@1 gate met.

## Techniques used (register K ids)

- K188, K189 (see `../../literature/register.md`)

## Data

- `results/e213.md`
- `results/e213b.md`
- `results/e213c.md`
- `results/regis_train_status.json`
- `figures/p18_registration.png`
- `figures/p18_arch.png`
- code: `../../training/models/p18_*.py`, `../../training/pipeline/p16_infer.py`, `../../evaluation/` (p18_*), launchers `../../temp/launchers/loop18/`
- raw outputs (arrays, checkpoints): `../../runs/`
