# E343 - E324 s on VAL (873 frames, 22 scenes): the arms against E312

Report only: nothing selected, tuned or re-read on this split. Control c = E312 snapped (its stored VAL card); w = E324 s snapped; p = the PRIMARY end-to-end read-out (attached ends + snap-nogeom wiring); k = the stack E338 (E324 s's pair head on the delivered graph of c2, snapped); snap_nogeom is reported only. The reading below is the rule of the card (`reading()`), no promotion of p or k is recommended whatever the numbers.

## reading (card E343 section 0)

**FULL RETENTION**

| seed | D = topo F1 w - E312 | lo | hi | D79 guards against E312 | dup_frac rise (D79 bound 0.02 decides; E324 VAL bound 0.01 printed only) |
|---|---|---|---|---|---|
| 0 | +0.015736 | +0.008701 | +0.023268 | hold | +0.0035 (ok / ok) |
| 1 | +0.014629 | +0.009505 | +0.019614 | hold | -0.0034 (ok / ok) |

Recommendation: E324 s may become the system of record in PROBLEM.md and the registry (the owner's decision, with the D94 / D95 grants)

## every arm against E312 snapped (D = arm - E312, paired 95 % scene bootstrap when E312's per-frame counts reproduced)

| arm | seed | topo F1 E312 | topo F1 arm | D | lanes found | junction F1 | dup_frac | F1@1.0 | vertices / 100 m | D79 guards |
|---|---|---|---|---|---|---|---|---|---|---|
| w (snapped) | 0 | 0.6823 | 0.6980 | +0.0157 [+0.0087, +0.0233] | +0.0231 [+0.0137, +0.0326] | +0.0043 [-0.0003, +0.0090] | +0.0035 [-0.0042, +0.0111] | +0.0093 [+0.0054, +0.0132] | -0.5795 [-0.7284, -0.4296] | hold |
| w (snapped) | 1 | 0.6864 | 0.7010 | +0.0146 [+0.0095, +0.0196] | +0.0172 [+0.0078, +0.0273] | +0.0088 [+0.0016, +0.0171] | -0.0034 [-0.0103, +0.0028] | +0.0079 [+0.0045, +0.0115] | -0.4730 [-0.6531, -0.2965] | hold |
| p (primary) | 0 | 0.6823 | 0.6911 | +0.0088 [+0.0003, +0.0169] | +0.0279 [+0.0195, +0.0373] | -0.0155 [-0.0241, -0.0064] | +0.0151 [+0.0084, +0.0222] | +0.0108 [+0.0067, +0.0150] | -0.3422 [-0.4947, -0.1941] | hold |
| p (primary) | 1 | 0.6864 | 0.6948 | +0.0084 [+0.0019, +0.0147] | +0.0229 [+0.0132, +0.0339] | -0.0094 [-0.0213, +0.0089] | +0.0036 [-0.0055, +0.0110] | +0.0102 [+0.0058, +0.0151] | -0.2611 [-0.4349, -0.0681] | hold |
| k (stack) | 0 | 0.6823 | 0.7066 | +0.0243 [+0.0149, +0.0334] | +0.0253 [+0.0129, +0.0346] | +0.0223 [+0.0127, +0.0317] | -0.0079 [-0.0222, +0.0043] | +0.0069 [+0.0008, +0.0123] | -0.6528 [-0.8904, -0.4127] | hold |
| k (stack) | 1 | 0.6864 | 0.7016 | +0.0152 [+0.0082, +0.0217] | +0.0262 [+0.0167, +0.0362] | +0.0167 [-0.0011, +0.0362] | +0.0031 [-0.0116, +0.0150] | +0.0103 [+0.0064, +0.0144] | -0.7719 [-1.1034, -0.4690] | hold |

## arms against each other (topo F1 difference, paired 95 % scene bootstrap)

| comparison | seed | topo F1 | lanes found | junction F1 | dup_frac |
|---|---|---|---|---|---|
| k - w (stack - snapped) | 0 | +0.0086 [+0.0037, +0.0145] | +0.0021 [-0.0051, +0.0097] | +0.0181 [+0.0093, +0.0274] | -0.0114 [-0.0212, -0.0021] |
| k - w (stack - snapped) | 1 | +0.0005 [-0.0044, +0.0058] | +0.0089 [+0.0036, +0.0156] | +0.0080 [-0.0083, +0.0236] | +0.0065 [-0.0034, +0.0152] |
| w - p (snapped - primary: the measured price of the end-to-end read-out) | 0 | +0.0069 [+0.0024, +0.0124] | -0.0048 [-0.0125, +0.0016] | +0.0198 [+0.0123, +0.0270] | -0.0116 [-0.0147, -0.0088] |
| w - p (snapped - primary: the measured price of the end-to-end read-out) | 1 | +0.0062 [+0.0031, +0.0093] | -0.0057 [-0.0102, -0.0001] | +0.0182 [+0.0055, +0.0279] | -0.0070 [-0.0111, -0.0036] |

E2E gap (named): arm w contains a rule step (the snap's wiring and keypoint geometry); arm p is the end-to-end read-out.

## snapped against its plain delivered graph (plain; connectors at p >= 0.1)

| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6259 | 0.6980 | +0.0721 [+0.0557, +0.0857] | +0.1340 [+0.1210, +0.1454] | +0.0520 [+0.0403, +0.0643] | -0.0565 [-0.0745, -0.0365] | +0.0479 [+0.0392, +0.0548] | -0.3135 [-0.6262, -0.0453] |
| 1 | 0.6249 | 0.7010 | +0.0761 [+0.0576, +0.0944] | +0.1455 [+0.1242, +0.1667] | +0.0737 [+0.0528, +0.1022] | -0.0489 [-0.0729, -0.0270] | +0.0603 [+0.0425, +0.0792] | -0.4533 [-0.9044, -0.0944] |

## snap_nogeom against its plain delivered graph (plain; connectors at p >= 0.1)

| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6259 | 0.6888 | +0.0629 [+0.0473, +0.0764] | +0.1355 [+0.1232, +0.1472] | +0.0365 [+0.0221, +0.0542] | -0.0521 [-0.0708, -0.0315] | +0.0476 [+0.0384, +0.0555] | -0.1740 [-0.4701, +0.0979] |
| 1 | 0.6249 | 0.6916 | +0.0667 [+0.0490, +0.0857] | +0.1476 [+0.1266, +0.1680] | +0.0552 [+0.0312, +0.0950] | -0.0436 [-0.0692, -0.0206] | +0.0608 [+0.0421, +0.0808] | -0.3373 [-0.7527, -0.0001] |

## primary against its plain delivered graph (plain; connectors at p >= 0.1)

| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6259 | 0.6911 | +0.0652 [+0.0514, +0.0777] | +0.1388 [+0.1240, +0.1541] | +0.0322 [+0.0199, +0.0452] | -0.0448 [-0.0630, -0.0250] | +0.0494 [+0.0406, +0.0571] | -0.0761 [-0.3776, +0.1879] |
| 1 | 0.6249 | 0.6948 | +0.0699 [+0.0527, +0.0875] | +0.1511 [+0.1303, +0.1719] | +0.0556 [+0.0323, +0.0942] | -0.0419 [-0.0678, -0.0177] | +0.0626 [+0.0455, +0.0811] | -0.2413 [-0.6445, +0.0932] |

## stack against its plain delivered graph (c2_plain; connectors at p >= 0.1)

| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6346 | 0.7066 | +0.0720 [+0.0563, +0.0858] | +0.1322 [+0.1207, +0.1426] | +0.0577 [+0.0447, +0.0701] | -0.0599 [-0.0900, -0.0301] | +0.0467 [+0.0382, +0.0550] | -0.0858 [-0.4213, +0.2017] |
| 1 | 0.6371 | 0.7016 | +0.0645 [+0.0485, +0.0804] | +0.1293 [+0.1126, +0.1464] | +0.0697 [+0.0506, +0.0918] | -0.0446 [-0.0642, -0.0277] | +0.0478 [+0.0362, +0.0588] | -0.2259 [-0.6049, +0.0915] |

## maneuver coverage (E310) - not verified against a stored card (none exists); its VAL value is a plausibility line

| seed | plain | p0.1s | p0.1an | c2 plain | c2 p0.1s |
|---|---|---|---|---|---|
| 0 | 52.5 % | 73.9 % | 75.7 % | 52.6 % | 74.1 % |
| 1 | 49.1 % | 73.5 % | 74.8 % | 52.8 % | 73.8 % |

## provenance

Device of the dumps and the scoring: cuda. Imported lane code: 71 files, digest 889d2d9387e0ba63; no change of those files during the run.

Environment: torch 2.8.0+cu129, numpy 1.26.4, shapely 2.0.7; 11 small files hashed (data/splits.json, val_tokens.json, poses_val.npz, run configs).
