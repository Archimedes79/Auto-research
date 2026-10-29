# Calibration audit (2026-10-29, layer 0-1, real VAL frames, CPU)

Script `temp/calib/check_calibration.py` (stride 8: 389 of 3,109 VAL keyframes, 82 scenes, 1,403 visible (>= 60 %) vehicle boxes),
raw output `calibration_raw.txt`. Request: "make sure calibration is right".

| check | method | result |
|---|---|---|
| A  geometry chain | global -> ego pose of the camera frame -> `T_cam_car` -> `K` against the nuScenes devkit (`get_sample_data` boxes in the camera frame, `view_points`) | max difference 8.5e-13 m (camera-frame centre), 2.1e-10 px |
| B  camera, input crop and lift pixel | probability mass of the cached (frozen) depth head inside the box's depth interval at the projected box centre, image shifted by (du, dv) px of the 704 x 256 input | peak at (0, 0): 0.666 (n = 760 boxes, 10-45 m); a shift of 16 px (one feature cell) costs 0.07-0.14, 48 px costs 0.35-0.5. The ridge is symmetric within +-8 px (the sweep step) |
| B2 depth bias | E[depth] - true centre depth, shift 0 | median -1.4 / -0.5 / -0.2 / -0.9 m at 10-20 / 20-30 / 30-40 / 40-50 m: the head predicts the visible surface, i.e. about half a car length in front of the centre, as expected |
| C  radar position | radar detections (all 5 radars, 0.5 s window, Doppler shift, moved to the camera time: the dataset's own `_radar`) inside the ground footprint (+ 0.5 m) of vehicle boxes, box shifted by (dx, dy) m | peak at (0, 0): 15,782 points; +-0.5 m costs 1-5 %, +-1.5 m costs 30-40 %. The plateau is wide (radar position noise), so the check resolves about 0.5 m |
| C2 radar time / Doppler | same, box moved along its annotated velocity by t s | peak at t = 0 (15,782); -0.2 s 15,066, +0.2 s 14,334, +-0.4 s 14,338 / 13,432 |

Reading: no calibration, ego-motion or time error was found at the resolution of the checks (camera: +-8 px of 704, i.e. +-0.5 feature cell;
radar: +-0.5 m, +-0.1 s). The camera branch is not weak because of a geometry error. Not checked: the other five cameras (not used),
lidar (not downloaded), radar extrinsics below 0.5 m / yaw below about 1 degree (the footprint check is too coarse; a lidar-free finer
check would need a radar-based tracker).
