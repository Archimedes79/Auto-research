# Board - arms against the base, all heads (evaluation/board.py, 2026-11-01)

Base `t8_ctl` (t8_ctl_s0, t8_ctl_s1) | GT signature 0.5_4_15efbde2d1 | split val | memory empty (cold) | champion file: proposal, not promoted

Base row = mean value; arm rows = change against the base mean. Markers: ^ better and v worse than the band in every seed, ~ inside the band, ? mean beyond the band but the seeds disagree. Lower is better: veh AVE.
Band per column (max of 0.02 and 2 x base seed spread): drivable 0.020, walkway 0.020, carpark 0.020, crossing 0.020, lane div. 0.020, road div. 0.020, mAP 0.033, veh AP@2 0.020, ped AP@2 0.020, veh AVE 0.028, drivable +0.5 s 0.020.
This table is evidence. The decision is the card's rule (`decide.py`).

| arm | state | maps: drivable | maps: walkway | maps: carpark | maps: crossing | maps: lane div. | maps: road div. | objects: mAP | objects: veh AP@2 | objects: ped AP@2 | motion: veh AVE | time: drivable +0.5 s | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t8_ctl (base) | 2 seed(s), 96a02be | 0.716 | 0.177 | 0.110 | 0.001 | 0.185 | 0.284 | 0.243 | 0.613 | 0.191 | 2.369 | 0.687 |  |
| t8_bevfeat | 2 of 2 seed(s), 96a02be, not recorded in the eval: GT signature | +0.090 ^ | +0.164 ^ | +0.060 ^ | +0.154 ^ | +0.163 ^ | +0.217 ^ | -0.134 v | -0.201 v | -0.190 v | +0.042 ? | +0.088 ^ | 7 better, 3 worse (mAP, veh AP@2, ped AP@2), 1 unclear |
| t7_res4 | 1 of 2 seed(s), 4719d1d, CODE DIFFERS (arm 4719d1d, base 96a02be), not recorded in the eval: GT signature | -0.017 ~ | -0.021 v | -0.006 ~ | -0.001 ~ | +0.035 ^ | +0.098 ^ | -0.025 ~ | -0.002 ~ | -0.013 ~ | +0.001 ~ | -0.013 ~ | 2 better, 1 worse (walkway) |
| t7_rgb | 1 of 2 seed(s), 4719d1d, CODE DIFFERS (arm 4719d1d, base 96a02be), not recorded in the eval: GT signature | +0.005 ~ | +0.000 ~ | +0.025 ^ | -0.001 ~ | +0.011 ~ | +0.053 ^ | -0.010 ~ | +0.003 ~ | -0.008 ~ | -0.023 ~ | +0.006 ~ | 2 better, 0 worse |
| t10_dual | 2 of 2 seed(s), 40648bd153, CODE DIFFERS (arm 40648bd153, 77aed2787c, base 96a02be) | +0.090 ^ | +0.170 ^ | +0.031 ^ | +0.162 ^ | +0.157 ^ | +0.218 ^ | -0.016 ~ | -0.001 ~ | -0.011 ~ | +0.011 ~ | +0.088 ^ | 7 better, 0 worse |
