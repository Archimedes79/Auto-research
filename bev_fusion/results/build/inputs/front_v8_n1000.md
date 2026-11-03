# Front-area comparison (x in [0, 50) m, +-32.3 deg), val frames, every frame from an empty memory (cold), 1000 frames of 82 scenes

AP by centre distance over 0.5 / 1 / 2 / 4 m (nuScenes style), mAP2 = mean of vehicle and pedestrian AP over the four distances; interval of the difference to `dualbev_front` = paired scene bootstrap (1000 draws, seed 0, 2.5 / 97.5 %).

- `dualbev_front`: temp/t16/soa/dualbev_front_v8_n1000.pkl (evaluation GT 0.5_4_6a7aa7864a)
- `t16_full_s0`: runs/t16_full_s0/eval/scene_records_n1000.pkl (evaluation GT 0.5_4_6a7aa7864a)
- `t17_data`: runs/t17_data/eval/scene_records_n1000.pkl (evaluation GT 0.5_4_6a7aa7864a)

| system | mAP2 | d vs ref [95% interval] | veh AP@0.5 | veh AP@1 | veh AP@2 | veh AP@4 | ped AP@0.5 | ped AP@1 | ped AP@2 | ped AP@4 |
|---|---|---|---|---|---|---|---|---|---|---|
| dualbev_front | 0.245 | - | 0.062 | 0.260 | 0.545 | 0.732 | 0.000 | 0.035 | 0.105 | 0.217 |
| t16_full_s0 | 0.442 | +0.197 [+0.178, +0.216] | 0.340 | 0.579 | 0.720 | 0.797 | 0.131 | 0.244 | 0.303 | 0.420 |
| t17_data | 0.444 | +0.200 [+0.180, +0.219] | 0.335 | 0.576 | 0.712 | 0.787 | 0.155 | 0.249 | 0.318 | 0.424 |

mAP2 by distance of the object (range bins of the metric; n/a: no ground truth in the bin):

| system | 0-20 m | 20-40 m | 40-50 m |
|---|---|---|---|
| dualbev_front | 0.442 | 0.205 | 0.061 |
| t16_full_s0 | 0.612 | 0.409 | 0.212 |
| t17_data | 0.605 | 0.415 | 0.215 |

GT boxes in the area (dualbev_front): vehicle 3060, pedestrian 1436.
