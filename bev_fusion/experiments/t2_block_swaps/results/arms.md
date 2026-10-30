> NOTE: the invalid `cam_off` arm (ran with the camera on) was removed from this table on 2026-10-30; the real radar-only arm is `t3_radar_only`. Camera-only = `radar_off`.

# t2_block_swaps: arms vs `ref_s0` (cold VAL, 1000 keyframes of 82 scenes, EMA, 3000 steps)

Noise band for the verdict: 0.054 (2 x seed spread, `ref_s1`). Difference = arm - `ref_s0`; interval = paired scene bootstrap (1000 draws).

| arm | mAP2 | veh AP@1 | ped AP@1 | veh AP@2 | ped AP@2 | mAP3 | veh AOE | veh AVE | drivable | walkway | lane div@0.5 | val loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_s0 | 0.339 | 0.468 | 0.141 | 0.606 | 0.204 | 0.277 | 1.13 | 2.39 | 0.666 | 0.191 | 0.181 | 11.60 |
| ref_s1 | 0.324 | 0.465 | 0.122 | 0.605 | 0.188 | 0.228 | 1.33 | 2.40 | 0.663 | 0.164 | 0.179 | 11.64 |
| t3_radar_only | 0.062 | 0.079 | 0.000 | 0.161 | 0.000 | 0.041 | 1.27 | 2.59 | 0.586 | 0.112 | 0.098 | 12.87 |
| t3_cam_zsum | 0.205 | 0.197 | 0.032 | 0.445 | 0.096 | 0.142 | 1.11 | 2.60 | 0.624 | 0.089 | 0.154 | 12.18 |
| t3_cam_adapter | 0.211 | 0.211 | 0.035 | 0.448 | 0.109 | 0.147 | 1.16 | 2.59 | 0.638 | 0.112 | 0.185 | 12.10 |
| radar_off | 0.210 | 0.209 | 0.033 | 0.448 | 0.104 | 0.147 | 1.12 | 2.53 | 0.637 | 0.107 | 0.164 | 12.07 |
| trunk_rand | 0.318 | 0.457 | 0.117 | 0.592 | 0.177 | 0.224 | 1.13 | 2.43 | 0.653 | 0.136 | 0.149 | 11.92 |
| lift_p | 0.290 | 0.426 | 0.094 | 0.549 | 0.148 | 0.194 | 1.30 | 2.36 | 0.677 | 0.170 | 0.167 | 11.81 |
| camtemp_none | 0.321 | 0.464 | 0.113 | 0.599 | 0.189 | 0.264 | 1.26 | 2.44 | 0.659 | 0.182 | 0.173 | 11.68 |
| fusion_gate | 0.330 | 0.470 | 0.126 | 0.613 | 0.188 | 0.226 | 1.10 | 2.39 | 0.668 | 0.185 | 0.190 | 11.69 |
| camtemp_gru | 0.327 | 0.465 | 0.118 | 0.614 | 0.179 | 0.221 | 1.19 | 2.38 | 0.675 | 0.200 | 0.192 | 11.60 |

## Differences and verdicts

| arm | mAP2 | veh AP@1 | ped AP@1 | veh AP@2 | ped AP@2 | mAP3 | veh AOE | veh AVE | drivable | walkway | lane div@0.5 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_s1 | -0.015 | -0.004 | -0.020* | -0.001 | -0.016* | -0.049 | +0.20 | +0.01 | -0.003 | -0.027* | -0.002 | TIE |
| t3_radar_only | -0.277 | -0.390* | -0.141* | -0.445* | -0.204* | -0.236 | +0.14 | +0.20 | -0.079* | -0.080* | -0.082* | WORSE: mAP2 (no bootstrap), veh AP@1, ped AP@1, veh AP@2, ped AP@2, drivable, walkway, lane div@0.5 |
| t3_cam_zsum | -0.134 | -0.271* | -0.110* | -0.161* | -0.108* | -0.135 | -0.02 | +0.21 | -0.042* | -0.102* | -0.027* | WORSE: mAP2 (no bootstrap), veh AP@1, ped AP@1, veh AP@2, ped AP@2, walkway |
| t3_cam_adapter | -0.128 | -0.257* | -0.107* | -0.158* | -0.095* | -0.130 | +0.03 | +0.20 | -0.028* | -0.079* | +0.004 | WORSE: mAP2 (no bootstrap), veh AP@1, ped AP@1, veh AP@2, ped AP@2, walkway |
| radar_off | -0.129 | -0.260* | -0.108* | -0.158* | -0.099* | -0.130 | -0.01 | +0.14 | -0.028* | -0.084* | -0.017* | WORSE: mAP2 (no bootstrap), veh AP@1, ped AP@1, veh AP@2, ped AP@2, walkway |
| trunk_rand | -0.021 | -0.011 | -0.024* | -0.014* | -0.026* | -0.053 | -0.00 | +0.04 | -0.012* | -0.055* | -0.032* | WORSE: walkway |
| lift_p | -0.048 | -0.042* | -0.048* | -0.057* | -0.056* | -0.083 | +0.16 | -0.03 | +0.011 | -0.021* | -0.013* | WORSE: veh AP@2, ped AP@2 |
| camtemp_none | -0.017 | -0.005 | -0.028* | -0.007 | -0.015 | -0.012 | +0.13 | +0.05 | -0.007* | -0.010* | -0.008* | TIE |
| fusion_gate | -0.009 | +0.001 | -0.016* | +0.007 | -0.016* | -0.051 | -0.03 | +0.00 | +0.002 | -0.006 | +0.010* | TIE |
| camtemp_gru | -0.011 | -0.003 | -0.024* | +0.008 | -0.025* | -0.056 | +0.06 | -0.01 | +0.009* | +0.008* | +0.011* | TIE |

`*` = paired bootstrap interval excludes 0. Velocity / heading errors (AVE, AOE) are lower-better and have no bootstrap; they are not used in the verdict.
