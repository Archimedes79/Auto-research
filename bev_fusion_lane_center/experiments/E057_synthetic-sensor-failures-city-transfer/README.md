# E057_synthetic-sensor-failures-city-transfer

**Synthetic sensor failures (night, blur, lost cameras, fog, radar, extrinsics) and city transfer**

| | |
|---|---|
| experiment id | E57 / E58 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | C - classical / reference: classical method, simulator, analysis, protocol or literature review |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| loop | loop05 |

## Result

F1 0.867 falls 70 % (night), 50 % (blur), 25 % (three cameras), 6 % (radar, fog); a new city costs 0.04 - 0.10

## Techniques used (register K ids)

- K81: benchmark of camera / LiDAR corruptions for online map models  (checked loop 7 (html))

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- code: removed in the clean-up of 2026-11-21, kept behind the git tag `archive/pre-cleanup-2026-11-20`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
