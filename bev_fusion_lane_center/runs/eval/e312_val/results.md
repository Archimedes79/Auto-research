# E312 on VAL (873 frames, 22 scenes) - winner against the plain delivered graph of B

Read-outs fixed on VAL: connectors at p >= 0.1 snapped (E311, the card of record of E312) and not snapped (the differentiable read-out); control = B + the close-node step `close:1.5@0.5`. Report only: nothing selected or tuned on this split.

## unsnapped against its plain delivered graph (plain; connectors at p >= 0.1)

| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6259 | 0.6437 | +0.0178 [+0.0085, +0.0280] | +0.1139 [+0.0972, +0.1308] | -0.0206 [-0.0368, -0.0037] | -0.0738 [-0.0921, -0.0528] | +0.0380 [+0.0298, +0.0453] | +0.4564 [+0.1466, +0.7447] |
| 1 | 0.6249 | 0.6481 | +0.0232 [+0.0068, +0.0429] | +0.1348 [+0.1109, +0.1603] | -0.0022 [-0.0237, +0.0274] | -0.0605 [-0.0851, -0.0372] | +0.0524 [+0.0351, +0.0721] | +0.1902 [-0.1712, +0.4734] |

## snapped against its plain delivered graph (plain; connectors at p >= 0.1)

| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6259 | 0.6823 | +0.0564 [+0.0423, +0.0694] | +0.1109 [+0.0942, +0.1287] | +0.0477 [+0.0371, +0.0597] | -0.0599 [-0.0765, -0.0411] | +0.0386 [+0.0304, +0.0454] | +0.2660 [-0.0292, +0.5037] |
| 1 | 0.6249 | 0.6864 | +0.0615 [+0.0445, +0.0795] | +0.1282 [+0.1069, +0.1521] | +0.0649 [+0.0471, +0.0872] | -0.0455 [-0.0652, -0.0247] | +0.0524 [+0.0362, +0.0709] | +0.0198 [-0.3736, +0.3094] |

## maneuver coverage (E310)

| seed | plain | p0.1 | p0.1s |
|---|---|---|---|
| 0 | 52.5 % | 71.3 % | 70.7 % |
| 1 | 49.1 % | 70.6 % | 70.1 % |

## provenance

Device of the dumps and the scoring: cpu. Imported lane code: 71 files, digest 889d2d9387e0ba63; no change of those files during the run.

Environment: torch 2.8.0+cu129, numpy 1.26.4, shapely 2.0.7; 7 small files hashed (data/splits.json, val_tokens.json, poses_val.npz, run configs).
