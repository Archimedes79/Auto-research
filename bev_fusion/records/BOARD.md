# Board - arms against the base, all heads (evaluation/board.py, 2026-10-31)

Base `t8_ctl` (t8_ctl_s0) | GT signature 0.5_4_15efbde2d1 | split val | memory empty (cold) | champion file: proposal, not promoted

Base row = mean value; arm rows = change against the base mean. Markers: ^ better and v worse than the band in every seed, ~ inside the band, ? mean beyond the band but the seeds disagree. Lower is better: veh AVE.
Band per column (max of 0.02 and 2 x base seed spread): drivable 0.020, walkway 0.020, carpark 0.020, crossing 0.020, lane div. 0.020, road div. 0.020, mAP 0.020, veh AP@2 0.020, ped AP@2 0.020, veh AVE 0.020, drivable +0.5 s 0.020.
Base has ONE seed: the band is the floor, not a measured spread. This table is evidence. The decision is the card's rule (`decide.py`).

| arm | state | maps: drivable | maps: walkway | maps: carpark | maps: crossing | maps: lane div. | maps: road div. | objects: mAP | objects: veh AP@2 | objects: ped AP@2 | motion: veh AVE | time: drivable +0.5 s | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t8_ctl (base) | 1 seed(s), 96a02be | 0.716 | 0.178 | 0.115 | 0.002 | 0.191 | 0.284 | 0.251 | 0.614 | 0.190 | 2.376 | 0.689 |  |
| t8_bevfeat | 2 of 2 seed(s), 96a02be, not recorded in the eval: GT signature | +0.090 ^ | +0.163 ^ | +0.056 ^ | +0.153 ^ | +0.158 ^ | +0.217 ^ | -0.142 v | -0.201 v | -0.189 v | +0.035 ? | +0.087 ^ | 7 better, 3 worse (mAP, veh AP@2, ped AP@2), 1 unclear |
| t7_res4 | 1 of 2 seed(s), 4719d1d, CODE DIFFERS (arm 4719d1d, base 96a02be), not recorded in the eval: GT signature | -0.018 ~ | -0.022 v | -0.010 ~ | -0.002 ~ | +0.030 ^ | +0.098 ^ | -0.034 v | -0.002 ~ | -0.012 ~ | -0.006 ~ | -0.014 ~ | 2 better, 2 worse (walkway, mAP) |
| t7_rgb | 1 of 2 seed(s), 4719d1d, CODE DIFFERS (arm 4719d1d, base 96a02be), not recorded in the eval: GT signature | +0.005 ~ | -0.001 ~ | +0.021 ^ | -0.002 ~ | +0.006 ~ | +0.053 ^ | -0.018 ~ | +0.003 ~ | -0.007 ~ | -0.030 ^ | +0.005 ~ | 3 better, 0 worse |
| t10_dual | 1 of 1 seed(s), 40648bd153, CODE DIFFERS (arm 40648bd153, base 96a02be) | +0.090 ^ | +0.161 ^ | +0.029 ^ | +0.152 ^ | +0.145 ^ | +0.215 ^ | -0.030 v | +0.000 ~ | -0.012 ~ | +0.009 ~ | +0.087 ^ | 7 better, 1 worse (mAP) |
