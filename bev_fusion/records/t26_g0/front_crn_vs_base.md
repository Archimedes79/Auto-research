# Front-area comparison (x in [0, 50) m, +-32.3 deg), val frames, every frame from an empty memory (cold), 1620 frames of 43 scenes

AP by centre distance over 0.5 / 1 / 2 / 4 m (nuScenes style), mAP2 = mean of vehicle and pedestrian AP over the four distances; interval of the difference to `crn` = paired scene bootstrap (1000 draws, seed 0, 2.5 / 97.5 %).

- `crn`: ../../../../home/ml/work/CRN/outputs/t26_crn/crn_front.pkl (evaluation GT 0.5_4_a107e0b4e8)
- `base_run`: temp/t26/base_run/eval/scene_records_nall.pkl (evaluation GT 0.5_4_a107e0b4e8)

| system | mAP2 | d vs ref [95% interval] | veh AP@0.5 | veh AP@1 | veh AP@2 | veh AP@4 | ped AP@0.5 | ped AP@1 | ped AP@2 | ped AP@4 |
|---|---|---|---|---|---|---|---|---|---|---|
| crn | 0.598 | - | 0.496 | 0.748 | 0.840 | 0.870 | 0.281 | 0.445 | 0.526 | 0.578 |
| base_run | 0.489 | -0.109 [-0.133, -0.087] | 0.430 | 0.672 | 0.778 | 0.826 | 0.147 | 0.249 | 0.339 | 0.469 |

mAP2 by distance of the object (range bins of the metric; n/a: no ground truth in the bin):

| system | 0-20 m | 20-40 m | 40-50 m |
|---|---|---|---|
| crn | 0.763 | 0.577 | 0.330 |
| base_run | 0.620 | 0.467 | 0.262 |

GT boxes in the area (crn): vehicle 4371, pedestrian 3276.
