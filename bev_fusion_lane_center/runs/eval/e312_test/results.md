# E312 on TEST (833 frames, 21 scenes) - winner against the plain delivered graph of B

Read-outs fixed on VAL: connectors at p >= 0.1 snapped (E311, the card of record of E312) and not snapped (the differentiable read-out); control = B + the close-node step `close:1.5@0.5`. Report only: nothing selected or tuned on this split.

## unsnapped (connectors at p >= 0.1)

| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5631 | 0.5864 | +0.0233 [+0.0121, +0.0363] | +0.1027 [+0.0897, +0.1161] | -0.0144 [-0.0287, +0.0016] | -0.0900 [-0.1204, -0.0610] | +0.0418 [+0.0339, +0.0500] | +0.5169 [+0.0849, +0.8781] |
| 1 | 0.5706 | 0.5947 | +0.0241 [+0.0105, +0.0396] | +0.1239 [+0.1095, +0.1403] | -0.0042 [-0.0220, +0.0130] | -0.0803 [-0.1153, -0.0429] | +0.0525 [+0.0427, +0.0624] | +0.0300 [-0.5369, +0.4646] |

## snapped (connectors at p >= 0.1)

| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5631 | 0.6264 | +0.0633 [+0.0519, +0.0751] | +0.1000 [+0.0875, +0.1137] | +0.0512 [+0.0340, +0.0714] | -0.0724 [-0.0995, -0.0462] | +0.0415 [+0.0347, +0.0489] | +0.3843 [-0.0507, +0.7211] |
| 1 | 0.5706 | 0.6326 | +0.0620 [+0.0492, +0.0755] | +0.1151 [+0.1007, +0.1322] | +0.0605 [+0.0415, +0.0843] | -0.0675 [-0.0974, -0.0332] | +0.0492 [+0.0394, +0.0599] | -0.0564 [-0.6114, +0.3458] |

## maneuver coverage (E310)

| seed | plain | connectors not snapped | connectors snapped |
|---|---|---|---|
| 0 | 47.7 % | 63.6 % | 63.2 % |
| 1 | 45.3 % | 64.2 % | 63.1 % |
