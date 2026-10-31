# Front-area comparison (x in [0, 50) m, +-32.3 deg), cold VAL, 1000 frames of 82 scenes

AP by centre distance over 0.5 / 1 / 2 / 4 m (nuScenes style), mAP2 = mean of vehicle and pedestrian AP over the four distances; interval of the difference to `t8_ctl_s0` = paired scene bootstrap (100 draws, 2.5 / 97.5 %).

| system | mAP2 | d vs ref [95% interval] | veh AP@0.5 | veh AP@1 | veh AP@2 | veh AP@4 | ped AP@0.5 | ped AP@1 | ped AP@2 | ped AP@4 |
|---|---|---|---|---|---|---|---|---|---|---|
| t8_ctl_s0 | 0.367 | - | 0.258 | 0.519 | 0.670 | 0.740 | 0.060 | 0.150 | 0.210 | 0.333 |
| t10_dual_s0 | 0.361 | -0.006 [-0.016, -0.000] | 0.254 | 0.517 | 0.670 | 0.746 | 0.047 | 0.136 | 0.199 | 0.318 |
| t8_bevfeat_s0 | 0.183 | -0.184 [-0.208, -0.162] | 0.101 | 0.313 | 0.455 | 0.574 | 0.000 | 0.000 | 0.001 | 0.021 |
| t8_bevfeat_s1 | 0.180 | -0.188 [-0.213, -0.165] | 0.093 | 0.299 | 0.447 | 0.571 | 0.000 | 0.000 | 0.002 | 0.026 |

mAP2 by distance of the object (range bins of the metric):

| system | 0-20 m | 20-40 m | 40-50 m |
|---|---|---|---|
| t8_ctl_s0 | 0.542 | 0.334 | 0.159 |
| t10_dual_s0 | 0.523 | 0.332 | 0.154 |
| t8_bevfeat_s0 | 0.253 | 0.184 | 0.096 |
| t8_bevfeat_s1 | 0.238 | 0.179 | 0.097 |

GT boxes in the area: vehicle 3060, pedestrian 1436.
