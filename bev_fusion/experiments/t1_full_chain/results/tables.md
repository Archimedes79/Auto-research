# t1 results: baseline `runs/t1_r0` (step 5000, window-only, plain heading head) vs combined `runs/t1_r3_combined` (step 6000)

VAL split, 3109 keyframes of 82 scenes, EMA weights, camera cells (maps), front field of view <= 60 m (boxes), 0.5 m grid; every number from `runs/<run>/eval/metrics.json`.

## 1 Maps at the current time (dt = 0)

| metric | prior map | baseline | combined | combined - baseline |
|---|---|---|---|---|
| drivable area IoU@0 | 0.429 | 0.699 | 0.696 | -0.003 |
| drivable area IoU@0.5 m | 0.464 | 0.747 | 0.744 | -0.003 |
| walkway IoU@0 | 0.000 | 0.278 | 0.261 | -0.017 |
| carpark IoU@0 | 0.000 | 0.201 | 0.186 | -0.015 |
| pedestrian crossing IoU@0 | 0.000 | 0.106 | 0.097 | -0.009 |
| lane-divider mask IoU@0.5 m | 0.000 | 0.327 | 0.306 | -0.021 |
| lane-divider mask IoU@1 m | 0.000 | 0.433 | 0.412 | -0.021 |
| road-divider mask IoU@1 m | 0.000 | 0.425 | 0.390 | -0.035 |
| lane-divider zero-crossing F1@0.5 m | - | 0.068 | 0.029 | -0.040 |
| road-divider zero-crossing F1@0.5 m | - | 0.190 | 0.070 | -0.120 |
| lane-divider distance-field error near the line (m, lower is better) | 1.332 | 0.691 | 0.737 | +0.046 |

## 2 Time offsets: the same maps decoded at t - 0.5 s, t, t + 0.5 s (drivable area IoU@0, vehicle AP@2 m)

| quantity | dt | baseline | combined | combined - baseline |
|---|---|---|---|---|
| drivable area IoU@0 | dt-0.50 | 0.724 | 0.711 | -0.014 |
| drivable area IoU@0 | dt+0.00 | 0.699 | 0.696 | -0.003 |
| drivable area IoU@0 | dt+0.50 | 0.683 | 0.680 | -0.003 |
| vehicle AP@2 m | dt-0.50 | 0.479 | 0.451 | -0.028 |
| vehicle AP@2 m, combined with refined boxes | dt-0.50 | 0.479 | 0.408 | -0.071 |
| vehicle AP@2 m | dt+0.00 | 0.643 | 0.632 | -0.011 |
| vehicle AP@2 m, combined with refined boxes | dt+0.00 | 0.643 | 0.596 | -0.047 |
| vehicle AP@2 m | dt+0.50 | 0.411 | 0.387 | -0.024 |
| vehicle AP@2 m, combined with refined boxes | dt+0.50 | 0.411 | 0.336 | -0.075 |
| mAP (3 classes) | dt-0.50 | 0.236 | 0.216 | -0.020 |
| mAP (3 classes) | dt+0.00 | 0.307 | 0.282 | -0.026 |
| mAP (3 classes) | dt+0.50 | 0.189 | 0.177 | -0.012 |

## 3 Objects at dt = 0 (nuScenes-style, centre-distance AP; TP errors at 2 m)

| metric | baseline | combined | combined, refined boxes | combined - baseline (plain) |
|---|---|---|---|---|
| mAP (3 classes, 4 thresholds) | 0.307 | 0.282 | 0.243 | -0.026 |
| vehicle AP@2 m | 0.643 | 0.632 | 0.596 | -0.011 |
| vehicle AP@4 m | 0.724 | 0.709 | 0.687 | -0.015 |
| vehicle centre error ATE (m) | 0.60 | 0.59 | 0.58 | -0.01 |
| vehicle size error ASE (1 - IoU) | 0.168 | 0.184 | 0.161 | +0.016 |
| vehicle heading error AOE (rad) | 1.78 | 0.59 | 0.59 | -1.19 |
| vehicle velocity error AVE (m/s) | 1.92 | 2.32 | 2.33 | +0.40 |
| pedestrian AP@2 m | 0.214 | 0.199 | 0.138 | -0.015 |
| pedestrian centre error ATE (m) | 0.69 | 0.68 | 0.67 | -0.01 |
| two-wheeler AP@2 m | 0.258 | 0.190 | 0.130 | -0.069 |

| coverage IoU@0 (dense object layer) | baseline | combined | combined - baseline |
|---|---|---|---|
| vehicle | 0.435 | 0.440 | +0.005 |
| pedestrian | 0.124 | 0.124 | -0.000 |
| two-wheeler | 0.175 | 0.118 | -0.057 |

## 4 Paired scene bootstrap (VAL scenes resampled with replacement, 1000 draws; 95 % interval of combined - baseline)

Maps at dt 0; boxes at dt 0 (plain boxes of both) and the refined boxes of the combined run against the baseline's.

```
paired scene bootstrap (1000 resamples), b - a; a = runs/t1_r0 (det), b = runs/t1_r3_combined (det); val; maps at dt 0, boxes at dt +0.0
drivable_area iou@0          a 0.6989  b 0.6957  b-a -0.0032  95% [-0.0087, +0.0033]
walkway iou@0                a 0.2781  b 0.2607  b-a -0.0174  95% [-0.0247, -0.0103]  (excludes 0)
lane_divider iou@0.5         a 0.3272  b 0.3060  b-a -0.0212  95% [-0.0310, -0.0117]  (excludes 0)
lane_divider:zc f1@0.5       a 0.0684  b 0.0286  b-a -0.0398  95% [-0.0543, -0.0255]  (excludes 0)
vehicle iou@0                a 0.4350  b 0.4402  b-a +0.0052  95% [+0.0008, +0.0105]  (excludes 0)
vehicle AP@2                 a 0.6431  b 0.6322  b-a -0.0108  95% [-0.0175, -0.0044]  (excludes 0)
pedestrian AP@2              a 0.2139  b 0.1991  b-a -0.0148  95% [-0.0271, -0.0026]  (excludes 0)
paired scene bootstrap (1000 resamples), b - a; a = runs/t1_r0 (det), b = runs/t1_r3_combined (detr); val; maps at dt 0, boxes at dt +0.0
drivable_area iou@0          a 0.6989  b 0.6957  b-a -0.0032  95% [-0.0087, +0.0033]
walkway iou@0                a 0.2781  b 0.2607  b-a -0.0174  95% [-0.0247, -0.0103]  (excludes 0)
lane_divider iou@0.5         a 0.3272  b 0.3060  b-a -0.0212  95% [-0.0310, -0.0117]  (excludes 0)
lane_divider:zc f1@0.5       a 0.0684  b 0.0286  b-a -0.0398  95% [-0.0543, -0.0255]  (excludes 0)
vehicle iou@0                a 0.4350  b 0.4402  b-a +0.0052  95% [+0.0008, +0.0105]  (excludes 0)
vehicle AP@2                 a 0.6431  b 0.5963  b-a -0.0468  95% [-0.0585, -0.0370]  (excludes 0)
pedestrian AP@2              a 0.2139  b 0.1375  b-a -0.0764  95% [-0.0991, -0.0570]  (excludes 0)
```

## 5 Does the carried world memory help? (last keyframe of val clips, warm = memory carried, cold = same frame from an empty memory)

```
runs/t1_r3_combined step 6000: warm vs cold memory on the last keyframe of 600 val clips of 3 keyframes (EMA, VAL)
dt+0.00
  drivable IoU@0 (in view)             warm   0.6962  cold   0.6948  warm - cold  +0.0014
  drivable IoU@0 (outside view)        warm   0.5668  cold   0.5463  warm - cold  +0.0205
  walkway IoU@0 (outside view)         warm   0.2545  cold   0.2396  warm - cold  +0.0150
  lane-div mask IoU@1 (outside view)   warm   0.3578  cold   0.3218  warm - cold  +0.0360
  drivable IoU@0 (camera or radar)     warm   0.6554  cold   0.6486  warm - cold  +0.0068
  walkway IoU@0                        warm   0.2620  cold   0.2626  warm - cold  -0.0006
  lane-divider mask IoU@1              warm   0.4200  cold   0.4159  warm - cold  +0.0042
  vehicle coverage IoU                 warm   0.4342  cold   0.4353  warm - cold  -0.0010
  vehicle AP@2                         warm   0.6275  cold   0.6259  warm - cold  +0.0015
  pedestrian AP@2                      warm   0.2040  cold   0.2049  warm - cold  -0.0009
  vehicle ATE                          warm   0.5810  cold   0.5824  warm - cold  -0.0014
  vehicle AVE                          warm   2.3464  cold   2.3183  warm - cold  +0.0281
  vehicle AP@2 refined                 warm   0.6007  cold   0.5913  warm - cold  +0.0094
  vehicle ATE refined                  warm   0.5739  cold   0.5697  warm - cold  +0.0042
dt+0.50
  drivable IoU@0 (in view)             warm   0.6815  cold   0.6790  warm - cold  +0.0025
  drivable IoU@0 (outside view)        warm   0.5834  cold   0.5693  warm - cold  +0.0141
  walkway IoU@0 (outside view)         warm   0.2511  cold   0.2382  warm - cold  +0.0129
  lane-div mask IoU@1 (outside view)   warm   0.3699  cold   0.3421  warm - cold  +0.0278
  drivable IoU@0 (camera or radar)     warm   0.6499  cold   0.6441  warm - cold  +0.0058
  walkway IoU@0                        warm   0.2315  cold   0.2333  warm - cold  -0.0017
  lane-divider mask IoU@1              warm   0.3986  cold   0.3957  warm - cold  +0.0028
  vehicle coverage IoU                 warm   0.3315  cold   0.3315  warm - cold  -0.0000
  vehicle AP@2                         warm   0.3908  cold   0.3844  warm - cold  +0.0064
  pedestrian AP@2                      warm   0.1367  cold   0.1487  warm - cold  -0.0120
  vehicle ATE                          warm   0.7666  cold   0.7615  warm - cold  +0.0051
  vehicle AVE                          warm   1.4744  cold   1.4747  warm - cold  -0.0003
  vehicle AP@2 refined                 warm   0.3524  cold   0.3344  warm - cold  +0.0180
  vehicle ATE refined                  warm   0.7617  cold   0.7565  warm - cold  +0.0052
dt-0.50
  drivable IoU@0 (in view)             warm   0.7147  cold   0.7109  warm - cold  +0.0039
  drivable IoU@0 (outside view)        warm   0.5326  cold   0.5122  warm - cold  +0.0203
  walkway IoU@0 (outside view)         warm   0.1971  cold   0.1818  warm - cold  +0.0153
  lane-div mask IoU@1 (outside view)   warm   0.3004  cold   0.2610  warm - cold  +0.0394
  drivable IoU@0 (camera or radar)     warm   0.6608  cold   0.6523  warm - cold  +0.0085
  walkway IoU@0                        warm   0.2425  cold   0.2431  warm - cold  -0.0007
  lane-divider mask IoU@1              warm   0.4300  cold   0.4242  warm - cold  +0.0058
  vehicle coverage IoU                 warm   0.3783  cold   0.3697  warm - cold  +0.0086
  vehicle AP@2                         warm   0.4838  cold   0.4527  warm - cold  +0.0311
  pedestrian AP@2                      warm   0.2277  cold   0.2022  warm - cold  +0.0255
  vehicle ATE                          warm   0.6916  cold   0.7132  warm - cold  -0.0216
  vehicle AVE                          warm   1.9306  cold   1.7902  warm - cold  +0.1404
  vehicle AP@2 refined                 warm   0.4367  cold   0.4090  warm - cold  +0.0277
  vehicle ATE refined                  warm   0.6754  cold   0.7059  warm - cold  -0.0305
```

## 6 Training and cost

- baseline: 5000 optimiser steps x 16 samples (4 x 4), lr 0.0003, kendall task weights, model options {'trunk_width': 0.5}; validation losses (VAL, all samples, dt-weighted): areas 1.064, lines 1.091, objects 10.259, motion 0.964, total 8.704
- combined: 6000 optimiser steps x 18 samples (2 x 3 clips of 3 keyframes), lr 0.0003, kendall task weights, model options {'trunk_width': 0.5, 'object_direction': True, 'refine': True}; validation losses (VAL, all samples, dt-weighted): areas 1.061, lines 1.129, objects 8.680, motion 1.035, total 7.678

| latency, median ms per frame, batch 1 | baseline | combined |
|---|---|---|
| chain at dt 0 | 43.4 | 43.9 |
| chain at dt -0.5 / 0 / +0.5 s | 62.6 | 62.8 |
| chain at dt 0 + decoded and refined boxes | - | 47.5 |
| frozen camera heads (ResNet-50 + FPN + DepthNet, cached in training) | 17.3 | 17.3 |

baseline: GPU NVIDIA GeForce RTX 3060, idle before the measurement: True (utilisation 0.0 %), the chain without the camera heads (their output is cached in training; at run time they add the last row), RTX 3060 12 GB, fp32, PyTorch eager

combined: GPU NVIDIA GeForce RTX 3060, idle before the measurement: True (utilisation 0.0 %), the chain without the camera heads (their output is cached in training; at run time they add the last row), RTX 3060 12 GB, fp32, PyTorch eager
