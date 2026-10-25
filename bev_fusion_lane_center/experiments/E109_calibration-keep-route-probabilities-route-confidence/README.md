# E109_calibration-keep-route-probabilities-route-confidence

**Calibration of keep and route probabilities; route confidence as trust signal**

| | |
|---|---|
| experiment id | E109 / E110 |
| theme | T7 - Planner interface, trust and the planner loss (loops 5 - 14) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP4 keep and verify (precision gates, verifiers, probabilities) |
| loop | loop12 - log: [../../experiment_execution/indexes/LOOP12_INDEX.md](../../experiment_execution/indexes/LOOP12_INDEX.md) |

## Result

route probability of A calibrated (ECE 0.006, at most 0.016 under failure), that of B over-predicts the route length by 28 %; route confidence AUROC 0.88 - 0.91

## Techniques used (register K ids)

- K104: calibration and temperature scaling  (checked loop 12 (abstract))
- K105: calibration under dataset shift  (checked loop 12 (abstract))
- K87: online map uncertainty consumed by a trajectory predictor  (checked loops 8, 12 (abstract))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `code/`: `_e109.sh` (reference copy of the scripts that ran it; the shared library is in `../../training`)
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
