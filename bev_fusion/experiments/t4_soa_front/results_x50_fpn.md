# Front-area comparison (x in [0, 50) m, +-32.3 deg), cold VAL, 1000 frames of 82 scenes

AP by centre distance over 0.5 / 1 / 2 / 4 m (nuScenes style), mAP2 = mean of vehicle and pedestrian AP over the four distances; interval of the difference to `ref_s0` = paired scene bootstrap (100 draws, 2.5 / 97.5 %).

| system | mAP2 | d vs ref [95% interval] | veh AP@0.5 | veh AP@1 | veh AP@2 | veh AP@4 | ped AP@0.5 | ped AP@1 | ped AP@2 | ped AP@4 |
|---|---|---|---|---|---|---|---|---|---|---|
| ref_s0 | 0.375 | - | 0.271 | 0.524 | 0.661 | 0.733 | 0.066 | 0.161 | 0.226 | 0.356 |
| ref_s1 | 0.359 | -0.016 [-0.023, -0.007] | 0.252 | 0.521 | 0.660 | 0.729 | 0.043 | 0.140 | 0.208 | 0.320 |
| fpn_order | 0.367 | -0.008 [-0.016, -0.003] | 0.264 | 0.530 | 0.669 | 0.743 | 0.047 | 0.144 | 0.209 | 0.324 |

mAP2 by distance of the object (range bins of the metric):

| system | 0-20 m | 20-40 m | 40-50 m |
|---|---|---|---|
| ref_s0 | 0.557 | 0.341 | 0.161 |
| ref_s1 | 0.519 | 0.333 | 0.159 |
| fpn_order | 0.531 | 0.335 | 0.160 |

GT boxes in the area: vehicle 3061, pedestrian 1435.
