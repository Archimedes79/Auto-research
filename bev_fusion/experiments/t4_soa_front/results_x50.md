# Front-area comparison (x in [0, 50) m, +-32.3 deg), cold VAL, 1000 frames of 82 scenes

AP by centre distance over 0.5 / 1 / 2 / 4 m (nuScenes style), mAP2 = mean of vehicle and pedestrian AP over the four distances; interval of the difference to `DualBEV-front` = paired scene bootstrap (100 draws, 2.5 / 97.5 %).

| system | mAP2 | d vs ref [95% interval] | veh AP@0.5 | veh AP@1 | veh AP@2 | veh AP@4 | ped AP@0.5 | ped AP@1 | ped AP@2 | ped AP@4 |
|---|---|---|---|---|---|---|---|---|---|---|
| DualBEV-front | 0.250 | - | 0.076 | 0.277 | 0.559 | 0.733 | 0.000 | 0.035 | 0.105 | 0.217 |
| ours-ref_s0 | 0.375 | +0.124 [+0.108, +0.136] | 0.271 | 0.524 | 0.661 | 0.733 | 0.066 | 0.161 | 0.226 | 0.356 |
| ours-ref_s1 | 0.359 | +0.109 [+0.093, +0.120] | 0.252 | 0.521 | 0.660 | 0.729 | 0.043 | 0.140 | 0.208 | 0.320 |
| ours-camera-only | 0.242 | -0.009 [-0.021, -0.002] | 0.065 | 0.256 | 0.516 | 0.685 | 0.005 | 0.042 | 0.122 | 0.243 |
| ours-ref-6000-steps | 0.386 | +0.135 [+0.118, +0.151] | 0.277 | 0.539 | 0.673 | 0.747 | 0.077 | 0.174 | 0.237 | 0.362 |
| ours-long-7000x2 | 0.400 | +0.150 [+0.133, +0.169] | 0.293 | 0.551 | 0.688 | 0.760 | 0.086 | 0.195 | 0.255 | 0.371 |
| ours-vel10 | 0.360 | +0.110 [+0.093, +0.123] | 0.248 | 0.525 | 0.670 | 0.743 | 0.051 | 0.134 | 0.200 | 0.310 |

mAP2 by distance of the object (range bins of the metric):

| system | 0-20 m | 20-40 m | 40-50 m |
|---|---|---|---|
| DualBEV-front | 0.447 | 0.211 | 0.063 |
| ours-ref_s0 | 0.557 | 0.341 | 0.161 |
| ours-ref_s1 | 0.519 | 0.333 | 0.159 |
| ours-camera-only | 0.439 | 0.204 | 0.054 |
| ours-ref-6000-steps | 0.572 | 0.355 | 0.158 |
| ours-long-7000x2 | 0.571 | 0.370 | 0.176 |
| ours-vel10 | 0.511 | 0.334 | 0.162 |

GT boxes in the area: vehicle 3061, pedestrian 1435.
