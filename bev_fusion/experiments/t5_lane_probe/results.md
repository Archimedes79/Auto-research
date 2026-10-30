# t5 lane probe - results (VAL: 300 keyframes of 75 scenes; TRAIN: 600 keyframes)

Divider IoU on the stride-4 image grid (1 cell = 4 x 4 input pixels) at tolerance 0 / 1 / 2 cells (evaluation/metrics.SegMeter); bands = flat-ground distance of the pixel; threshold chosen on TRAIN frames; difference to `ctx16` with the paired scene bootstrap (1000 draws, 95 % interval) on IoU@1.

| arm | features | IoU@0 | IoU@1 | IoU@2 | IoU@1 2-20 m | 20-40 m | 40-60 m | d IoU@1 vs ctx16 [95 %] | thr | train IoU@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| prior | per-pixel prior (null) | 0.064 | 0.135 | 0.183 | 0.000 | 0.146 | 0.168 | -0.036 [-0.050, -0.022] | 0.5 | 0.092 |
| ctx16 | context 80 ch, stride 16 (lift input) | 0.086 | 0.171 | 0.245 | 0.089 | 0.187 | 0.216 | - | 0.5 | 0.157 |
| fpn16 | neck 512 ch, stride 16 | 0.094 | 0.190 | 0.269 | 0.131 | 0.214 | 0.219 | +0.019 [+0.009, +0.029] | 0.5 | 0.226 |
| res8 | ResNet layer 2, stride 8 | 0.088 | 0.189 | 0.278 | 0.158 | 0.228 | 0.204 | +0.018 [+0.003, +0.031] | 0.6 | 0.265 |
| res4 | ResNet layer 1, stride 4 | 0.066 | 0.148 | 0.222 | 0.130 | 0.188 | 0.165 | -0.023 [-0.038, -0.007] | 0.5 | 0.176 |
| rgb | RGB CNN from scratch (control) | 0.050 | 0.112 | 0.169 | 0.104 | 0.142 | 0.105 | -0.059 [-0.079, -0.041] | 0.4 | 0.137 |

Post hoc (added after the run, because the labels miss much visible paint; NOT pre-registered): recall of the labelled line cells at 1 cell over all pixels, precision at 1 cell only inside a corridor of 4 cells around the labelled lines (unlabelled paint further away is not counted), their F1; same threshold, heads, frames; difference of F1 to `ctx16` with the paired scene bootstrap.

| arm | recall@1 | corridor precision@1 | F1 | d F1 vs ctx16 [95 %] |
|---|---|---|---|---|
| prior | 0.327 | 0.523 | 0.402 | -0.006 [-0.036, +0.028] |
| ctx16 | 0.361 | 0.471 | 0.409 | - |
| fpn16 | 0.414 | 0.499 | 0.453 | +0.044 [+0.030, +0.060] |
| res8 | 0.321 | 0.500 | 0.391 | -0.018 [-0.040, +0.006] |
| res4 | 0.285 | 0.473 | 0.356 | -0.053 [-0.078, -0.026] |
| rgb | 0.208 | 0.443 | 0.283 | -0.126 [-0.157, -0.090] |

GT divider cells per val frame: mean 138 of 11264; frames without a divider: 10.
