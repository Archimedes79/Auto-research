# t2_block_swaps: arms vs `ref_s0` (cold VAL, 1000 keyframes of 82 scenes, EMA, 3000 steps)

Noise band for the verdict: 0.054 (2 x seed spread, `ref_s1`). Difference = arm - `ref_s0`; interval = paired scene bootstrap (1000 draws).

| arm | mAP2 | veh AP@1 | ped AP@1 | veh AP@2 | ped AP@2 | mAP3 | veh AOE | veh AVE | drivable | walkway | lane div@0.5 | val loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_s0 | 0.339 | 0.468 | 0.141 | 0.606 | 0.204 | 0.277 | 1.13 | 2.39 | 0.666 | 0.191 | 0.181 | 11.60 |
| ref_s1 | 0.324 | 0.465 | 0.122 | 0.605 | 0.188 | 0.228 | 1.33 | 2.40 | 0.663 | 0.164 | 0.179 | 11.64 |
| radar_off | 0.210 | 0.209 | 0.033 | 0.448 | 0.104 | 0.147 | 1.12 | 2.53 | 0.637 | 0.107 | 0.164 | 12.07 |
| cam_off | 0.333 | 0.473 | 0.138 | 0.610 | 0.195 | 0.247 | 1.13 | 2.38 | 0.653 | 0.174 | 0.171 | 11.68 |
| trunk_rand | 0.318 | 0.457 | 0.117 | 0.592 | 0.177 | 0.224 | 1.13 | 2.43 | 0.653 | 0.136 | 0.149 | 11.92 |
| lift_p | 0.290 | 0.426 | 0.094 | 0.549 | 0.148 | 0.194 | 1.30 | 2.36 | 0.677 | 0.170 | 0.167 | 11.81 |
| camtemp_none | 0.321 | 0.464 | 0.113 | 0.599 | 0.189 | 0.264 | 1.26 | 2.44 | 0.659 | 0.182 | 0.173 | 11.68 |

## Differences and verdicts

| arm | mAP2 | veh AP@1 | ped AP@1 | veh AP@2 | ped AP@2 | mAP3 | veh AOE | veh AVE | drivable | walkway | lane div@0.5 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_s1 | -0.015 | -0.004 | -0.020* | -0.001 | -0.016* | -0.049 | +0.20 | +0.01 | -0.003 | -0.027* | -0.002 | TIE |
| radar_off | -0.129 | -0.260* | -0.108* | -0.158* | -0.099* | -0.130 | -0.01 | +0.14 | -0.028* | -0.084* | -0.017* | WORSE: mAP2 (no bootstrap), veh AP@1, ped AP@1, veh AP@2, ped AP@2, walkway |
| cam_off | -0.006 | +0.005 | -0.004 | +0.004 | -0.009 | -0.029 | -0.01 | -0.01 | -0.013* | -0.017* | -0.010* | TIE |
| trunk_rand | -0.021 | -0.011 | -0.024* | -0.014* | -0.026* | -0.053 | -0.00 | +0.04 | -0.012* | -0.055* | -0.032* | WORSE: walkway |
| lift_p | -0.048 | -0.042* | -0.048* | -0.057* | -0.056* | -0.083 | +0.16 | -0.03 | +0.011 | -0.021* | -0.013* | WORSE: veh AP@2, ped AP@2 |
| camtemp_none | -0.017 | -0.005 | -0.028* | -0.007 | -0.015 | -0.012 | +0.13 | +0.05 | -0.007* | -0.010* | -0.008* | TIE |

`*` = paired bootstrap interval excludes 0. Velocity / heading errors (AVE, AOE) are lower-better and have no bootstrap; they are not used in the verdict.
