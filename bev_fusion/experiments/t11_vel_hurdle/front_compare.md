# Front-area comparison (x in [0, 50) m, +-32.3 deg), cold VAL, 1000 frames of 82 scenes

AP by centre distance over 0.5 / 1 / 2 / 4 m (nuScenes style), mAP2 = mean of vehicle and pedestrian AP over the four distances; interval of the difference to `t10_dual_s0` = paired scene bootstrap (100 draws, 2.5 / 97.5 %).

| system | mAP2 | d vs ref [95% interval] | veh AP@0.5 | veh AP@1 | veh AP@2 | veh AP@4 | ped AP@0.5 | ped AP@1 | ped AP@2 | ped AP@4 |
|---|---|---|---|---|---|---|---|---|---|---|
| t10_dual_s0 | 0.361 | - | 0.254 | 0.517 | 0.670 | 0.746 | 0.047 | 0.136 | 0.199 | 0.318 |
| t10_dual_s1 | 0.364 | +0.003 [-0.003, +0.009] | 0.265 | 0.524 | 0.666 | 0.742 | 0.050 | 0.144 | 0.202 | 0.317 |
| B1 | 0.346 | -0.015 [-0.020, -0.007] | 0.204 | 0.472 | 0.627 | 0.722 | 0.055 | 0.147 | 0.211 | 0.332 |
| B2 | 0.347 | -0.014 [-0.020, -0.006] | 0.207 | 0.468 | 0.629 | 0.711 | 0.058 | 0.152 | 0.215 | 0.334 |

mAP2 by distance of the object (range bins of the metric):

| system | 0-20 m | 20-40 m | 40-50 m |
|---|---|---|---|
| t10_dual_s0 | 0.523 | 0.332 | 0.154 |
| t10_dual_s1 | 0.533 | 0.338 | 0.156 |
| B1 | 0.525 | 0.312 | 0.134 |
| B2 | 0.528 | 0.309 | 0.134 |

GT boxes in the area: vehicle 3060, pedestrian 1436.
