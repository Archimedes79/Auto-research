# t1 results: baseline `runs/t1_r0` (step 5000, window-only, plain heading head) vs combined `runs/t1_r4_soa` (step 7000)

VAL split, 3109 keyframes of 82 scenes, EMA weights, camera cells (maps), front field of view <= 60 m (boxes), 0.5 m grid; every number from `runs/<run>/eval/metrics.json`.

## 1 Maps at the current time (dt = 0)

| metric | prior map | baseline | combined | combined - baseline |
|---|---|---|---|---|
| drivable area IoU@0 | 0.429 | 0.699 | 0.698 | -0.001 |
| drivable area IoU@0.5 m | 0.464 | 0.747 | 0.746 | -0.001 |
| walkway IoU@0 | 0.000 | 0.278 | 0.244 | -0.034 |
| carpark IoU@0 | 0.000 | 0.201 | 0.162 | -0.038 |
| pedestrian crossing IoU@0 | 0.000 | 0.106 | 0.093 | -0.013 |
| lane-divider mask IoU@0.5 m | 0.000 | 0.327 | 0.274 | -0.053 |
| lane-divider mask IoU@1 m | 0.000 | 0.433 | 0.386 | -0.047 |
| road-divider mask IoU@1 m | 0.000 | 0.425 | 0.396 | -0.029 |
| lane-divider zero-crossing F1@0.5 m | - | 0.068 | 0.028 | -0.041 |
| road-divider zero-crossing F1@0.5 m | - | 0.190 | 0.107 | -0.083 |
| lane-divider distance-field error near the line (m, lower is better) | 1.332 | 0.691 | 0.753 | +0.062 |

## 2 Time offsets: the same maps decoded at t - 0.5 s, t, t + 0.5 s (drivable area IoU@0, vehicle AP@2 m)

| quantity | dt | baseline | combined | combined - baseline |
|---|---|---|---|---|
| drivable area IoU@0 | dt-0.50 | 0.724 | 0.709 | -0.016 |
| drivable area IoU@0 | dt+0.00 | 0.699 | 0.698 | -0.001 |
| drivable area IoU@0 | dt+0.50 | 0.683 | 0.670 | -0.013 |
| vehicle AP@2 m | dt-0.50 | 0.479 | 0.467 | -0.012 |
| vehicle AP@2 m | dt+0.00 | 0.643 | 0.627 | -0.016 |
| vehicle AP@2 m | dt+0.50 | 0.411 | 0.399 | -0.012 |
| mAP (3 classes) | dt-0.50 | 0.236 | 0.243 | +0.008 |
| mAP (3 classes) | dt+0.00 | 0.307 | 0.309 | +0.001 |
| mAP (3 classes) | dt+0.50 | 0.189 | 0.193 | +0.005 |

## 3 Objects at dt = 0 (nuScenes-style, centre-distance AP; TP errors at 2 m)

| metric | baseline | combined | combined, refined boxes | combined - baseline (plain) |
|---|---|---|---|---|
| mAP (3 classes, 4 thresholds) | 0.307 | 0.309 | - | +0.001 |
| vehicle AP@2 m | 0.643 | 0.627 | - | -0.016 |
| vehicle AP@4 m | 0.724 | 0.704 | - | -0.020 |
| vehicle centre error ATE (m) | 0.60 | 0.62 | - | +0.02 |
| vehicle size error ASE (1 - IoU) | 0.168 | 0.165 | - | -0.002 |
| vehicle heading error AOE (rad) | 1.78 | 0.64 | - | -1.14 |
| vehicle velocity error AVE (m/s) | 1.92 | 1.82 | - | -0.10 |
| pedestrian AP@2 m | 0.214 | 0.228 | - | +0.014 |
| pedestrian centre error ATE (m) | 0.69 | 0.69 | - | -0.00 |
| two-wheeler AP@2 m | 0.258 | 0.265 | - | +0.007 |

| coverage IoU@0 (dense object layer) | baseline | combined | combined - baseline |
|---|---|---|---|
| vehicle | 0.435 | 0.434 | -0.001 |
| pedestrian | 0.124 | 0.132 | +0.008 |
| two-wheeler | 0.175 | 0.172 | -0.003 |

## 4 Paired scene bootstrap (VAL scenes resampled with replacement, 1000 draws; 95 % interval of combined - baseline)

Maps at dt 0; boxes at dt 0 (plain boxes of both) and the refined boxes of the combined run against the baseline's.

```
paired scene bootstrap (1000 resamples), b - a; a = runs/t1_r0 (det), b = runs/t1_r4_soa (det); val; maps at dt 0, boxes at dt +0.0
drivable_area iou@0          a 0.6989  b 0.6979  b-a -0.0010  95% [-0.0075, +0.0062]
walkway iou@0                a 0.2781  b 0.2442  b-a -0.0340  95% [-0.0442, -0.0233]  (excludes 0)
lane_divider iou@0.5         a 0.3272  b 0.2737  b-a -0.0535  95% [-0.0717, -0.0349]  (excludes 0)
lane_divider:zc f1@0.5       a 0.0684  b 0.0276  b-a -0.0408  95% [-0.0514, -0.0306]  (excludes 0)
vehicle iou@0                a 0.4350  b 0.4344  b-a -0.0005  95% [-0.0052, +0.0043]
vehicle AP@2                 a 0.6431  b 0.6266  b-a -0.0164  95% [-0.0243, -0.0092]  (excludes 0)
pedestrian AP@2              a 0.2139  b 0.2280  b-a +0.0140  95% [-0.0039, +0.0302]
```

## 5 Does the carried world memory help? (last keyframe of val clips, warm = memory carried, cold = same frame from an empty memory)

```
runs/t1_r4_soa step 7000: warm vs cold memory on the last keyframe of 600 val clips of 3 keyframes (EMA, VAL)
dt+0.00
  drivable IoU@0 (in view)             warm   0.6964  cold   0.6972  warm - cold  -0.0008
  drivable IoU@0 (outside view)        warm   0.5620  cold   0.5342  warm - cold  +0.0278
  walkway IoU@0 (outside view)         warm   0.2252  cold   0.2162  warm - cold  +0.0090
  lane-div mask IoU@1 (outside view)   warm   0.3194  cold   0.3046  warm - cold  +0.0148
  drivable IoU@0 (camera or radar)     warm   0.6544  cold   0.6474  warm - cold  +0.0070
  walkway IoU@0                        warm   0.2485  cold   0.2480  warm - cold  +0.0004
  lane-divider mask IoU@1              warm   0.3944  cold   0.3878  warm - cold  +0.0066
  vehicle coverage IoU                 warm   0.4257  cold   0.4278  warm - cold  -0.0021
  vehicle AP@2                         warm   0.6281  cold   0.6219  warm - cold  +0.0062
  pedestrian AP@2                      warm   0.2365  cold   0.2339  warm - cold  +0.0026
  vehicle ATE                          warm   0.6228  cold   0.6232  warm - cold  -0.0004
  vehicle AVE                          warm   1.8157  cold   1.8037  warm - cold  +0.0119
dt+0.50
  drivable IoU@0 (in view)             warm   0.6676  cold   0.6695  warm - cold  -0.0019
  drivable IoU@0 (outside view)        warm   0.5656  cold   0.5465  warm - cold  +0.0192
  walkway IoU@0 (outside view)         warm   0.2382  cold   0.2315  warm - cold  +0.0067
  lane-div mask IoU@1 (outside view)   warm   0.3360  cold   0.3250  warm - cold  +0.0110
  drivable IoU@0 (camera or radar)     warm   0.6349  cold   0.6310  warm - cold  +0.0039
  walkway IoU@0                        warm   0.2288  cold   0.2280  warm - cold  +0.0008
  lane-divider mask IoU@1              warm   0.3768  cold   0.3700  warm - cold  +0.0067
  vehicle coverage IoU                 warm   0.3337  cold   0.3374  warm - cold  -0.0037
  vehicle AP@2                         warm   0.4017  cold   0.3942  warm - cold  +0.0074
  pedestrian AP@2                      warm   0.1719  cold   0.1710  warm - cold  +0.0009
  vehicle ATE                          warm   0.7690  cold   0.7801  warm - cold  -0.0112
  vehicle AVE                          warm   1.1900  cold   1.2549  warm - cold  -0.0650
dt-0.50
  drivable IoU@0 (in view)             warm   0.7095  cold   0.7079  warm - cold  +0.0016
  drivable IoU@0 (outside view)        warm   0.5244  cold   0.4969  warm - cold  +0.0275
  walkway IoU@0 (outside view)         warm   0.1742  cold   0.1660  warm - cold  +0.0082
  lane-div mask IoU@1 (outside view)   warm   0.2666  cold   0.2430  warm - cold  +0.0236
  drivable IoU@0 (camera or radar)     warm   0.6547  cold   0.6463  warm - cold  +0.0083
  walkway IoU@0                        warm   0.2367  cold   0.2368  warm - cold  -0.0000
  lane-divider mask IoU@1              warm   0.4041  cold   0.3962  warm - cold  +0.0079
  vehicle coverage IoU                 warm   0.3841  cold   0.3794  warm - cold  +0.0047
  vehicle AP@2                         warm   0.4963  cold   0.4616  warm - cold  +0.0347
  pedestrian AP@2                      warm   0.2594  cold   0.2293  warm - cold  +0.0301
  vehicle ATE                          warm   0.7237  cold   0.7370  warm - cold  -0.0133
  vehicle AVE                          warm   1.3884  cold   1.3209  warm - cold  +0.0676
```

## 6 Training and cost

- baseline: 5000 optimiser steps x 16 samples (4 x 4), lr 0.0003, kendall task weights, model options {'trunk_width': 0.5}; validation losses (VAL, all samples, dt-weighted): areas 1.064, lines 1.091, objects 10.259, motion 0.964, total 8.704
- combined: 7000 optimiser steps x 18 samples (3 x 2 clips of 3 keyframes), lr 0.0003, kendall task weights, model options {'trunk_width': 1.0, 'trunk_detail': 32, 'trunk_pretrained': 'models/pretrained/dualbev-r50-cbgs.pth', 'c_fuse': 80, 'object_direction': True, 'refine': False, 'head_channels': 96}; validation losses (VAL, all samples, dt-weighted): areas 1.077, lines 1.146, objects 11.209, motion 0.907, total 8.275

| latency, median ms per frame, batch 1 | baseline | combined |
|---|---|---|
| chain at dt 0 | 43.4 | 50.7 |
| chain at dt -0.5 / 0 / +0.5 s | 62.6 | 68.2 |
| frozen camera heads (ResNet-50 + FPN + DepthNet, cached in training) | 17.3 | 17.2 |

baseline: GPU NVIDIA GeForce RTX 3060, idle before the measurement: True (utilisation 0.0 %), the chain without the camera heads (their output is cached in training; at run time they add the last row), RTX 3060 12 GB, fp32, PyTorch eager

combined: GPU NVIDIA GeForce RTX 3060, idle before the measurement: True (utilisation 0.0 %), the chain without the camera heads (their output is cached in training; at run time they add the last row), RTX 3060 12 GB, fp32, PyTorch eager
