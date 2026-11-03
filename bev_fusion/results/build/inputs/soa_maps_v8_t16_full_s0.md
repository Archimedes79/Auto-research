# Map layers: runs/t16_full_s0 (ours) against GaussianCaR, GT v8 (0.5_4_6a7aa7864a), VAL

384 val frames of 41 scenes (the 384 of the 400 frames of the old scoring that are samples of this val set (16 are the first / last keyframe of a scene)), camera cells only (the GT camera mask, as `seg_fov`), checkpoint ckpt_00006000.pt step 6000 (EMA), memory empty before every frame, dt 0. VAL, not TEST. The denominator of each number is the number of GT cells (`n_gt`, summed over the frames) in the table. GaussianCaR: 6 cameras + radar, 360 degrees, trained for the nuScenes maps (its cached logits, forward offset 0.5 m); ours: front camera + radar. Prior: per-cell GT mean of 500 train frames (null). Tolerance t: a cell counts as right if the other map has a cell within t metres (iou@0 = plain IoU).

## Table 1. Prediction threshold 0.5, all ranges (the numbers to quote)

| layer | metric | n_gt cells | prior | ours | GaussianCaR | ours / GaussianCaR | ours (record, 1000 frames, 82 scenes) |
|---|---|---|---|---|---|---|---|
| drivable_area | iou@0 | 1311823 | 0.439 | 0.818 | 0.858 | 0.95 | 0.779 |
| drivable_area | iou@0.5 | 1311823 | 0.479 | 0.872 | 0.912 | 0.96 | 0.827 |
| drivable_area | iou@1 | 1311823 | 0.516 | 0.903 | 0.936 | 0.96 | 0.858 |
| ped_crossing | iou@0 | 73059 | 0.000 | 0.262 | 0.528 | 0.50 | 0.256 |
| ped_crossing | iou@0.5 | 73059 | 0.000 | 0.334 | 0.650 | 0.51 | 0.323 |
| ped_crossing | iou@1 | 73059 | 0.000 | 0.401 | 0.720 | 0.56 | 0.384 |
| walkway | iou@0 | 323589 | 0.000 | 0.417 | 0.594 | 0.70 | 0.413 |
| walkway | iou@0.5 | 323589 | 0.000 | 0.520 | 0.733 | 0.71 | 0.502 |
| walkway | iou@1 | 323589 | 0.000 | 0.594 | 0.799 | 0.74 | 0.565 |
| carpark_area | iou@0 | 31512 | 0.000 | 0.468 | 0.656 | 0.71 | 0.274 |
| carpark_area | iou@0.5 | 31512 | 0.000 | 0.616 | 0.857 | 0.72 | 0.340 |
| carpark_area | iou@1 | 31512 | 0.000 | 0.687 | 0.916 | 0.75 | 0.376 |
| lane_divider | iou@0 | 45885 | 0.000 | 0.200 | 0.281 | 0.71 | 0.189 |
| lane_divider | iou@0.5 | 45885 | 0.000 | 0.455 | 0.690 | 0.66 | 0.429 |
| lane_divider | iou@1 | 45885 | 0.000 | 0.583 | 0.833 | 0.70 | 0.549 |
| road_divider | iou@0 | 11457 | 0.000 | 0.252 | 0.258 | 0.97 | 0.244 |
| road_divider | iou@0.5 | 11457 | 0.000 | 0.544 | 0.614 | 0.89 | 0.506 |
| road_divider | iou@1 | 11457 | 0.000 | 0.612 | 0.756 | 0.81 | 0.572 |

## Table 2. Best of the thresholds {0.3, 0.5, 0.7} per layer and metric, chosen on these frames (IN-SAMPLE, for the model trained with another loss)

| layer | metric | n_gt cells | prior | ours | GaussianCaR | ours / GaussianCaR |
|---|---|---|---|---|---|---|
| drivable_area | iou@0 | 1311823 | 0.504 | 0.818 (thr 0.5) | 0.858 (thr 0.5) | 0.95 |
| drivable_area | iou@0.5 | 1311823 | 0.542 | 0.872 (thr 0.5) | 0.912 (thr 0.5) | 0.96 |
| drivable_area | iou@1 | 1311823 | 0.576 | 0.903 (thr 0.5) | 0.936 (thr 0.5) | 0.96 |
| ped_crossing | iou@0 | 73059 | 0.000 | 0.262 (thr 0.5) | 0.528 (thr 0.5) | 0.50 |
| ped_crossing | iou@0.5 | 73059 | 0.000 | 0.334 (thr 0.5) | 0.650 (thr 0.5) | 0.51 |
| ped_crossing | iou@1 | 73059 | 0.000 | 0.401 (thr 0.5) | 0.720 (thr 0.5) | 0.56 |
| walkway | iou@0 | 323589 | 0.000 | 0.428 (thr 0.3) | 0.594 (thr 0.5) | 0.72 |
| walkway | iou@0.5 | 323589 | 0.000 | 0.535 (thr 0.3) | 0.733 (thr 0.5) | 0.73 |
| walkway | iou@1 | 323589 | 0.000 | 0.618 (thr 0.3) | 0.799 (thr 0.5) | 0.77 |
| carpark_area | iou@0 | 31512 | 0.000 | 0.473 (thr 0.7) | 0.671 (thr 0.7) | 0.70 |
| carpark_area | iou@0.5 | 31512 | 0.000 | 0.624 (thr 0.7) | 0.857 (thr 0.5) | 0.73 |
| carpark_area | iou@1 | 31512 | 0.000 | 0.693 (thr 0.7) | 0.916 (thr 0.5) | 0.76 |
| lane_divider | iou@0 | 45885 | 0.000 | 0.202 (thr 0.7) | 0.367 (thr 0.7) | 0.55 |
| lane_divider | iou@0.5 | 45885 | 0.000 | 0.458 (thr 0.7) | 0.763 (thr 0.7) | 0.60 |
| lane_divider | iou@1 | 45885 | 0.000 | 0.584 (thr 0.7) | 0.833 (thr 0.5) | 0.70 |
| road_divider | iou@0 | 11457 | 0.000 | 0.253 (thr 0.7) | 0.334 (thr 0.7) | 0.76 |
| road_divider | iou@0.5 | 11457 | 0.000 | 0.545 (thr 0.7) | 0.683 (thr 0.7) | 0.80 |
| road_divider | iou@1 | 11457 | 0.000 | 0.612 (thr 0.5) | 0.756 (thr 0.5) | 0.81 |

## Table 3. By range, threshold 0.5 (ours / GaussianCaR / ratio; n_gt of the bin in brackets)

| layer | metric | 0-20 m | 20-40 m | 40-60 m |
|---|---|---|---|---|
| drivable_area | iou@0 | 0.919 / 0.940 / 0.98 (219414) | 0.835 / 0.898 / 0.93 (528211) | 0.766 / 0.793 / 0.97 (564198) |
| drivable_area | iou@0.5 | 0.963 / 0.986 / 0.98 (219414) | 0.889 / 0.952 / 0.93 (528211) | 0.823 / 0.850 / 0.97 (564198) |
| ped_crossing | iou@0 | 0.342 / 0.704 / 0.49 (8233) | 0.285 / 0.648 / 0.44 (33205) | 0.220 / 0.378 / 0.58 (31621) |
| ped_crossing | iou@0.5 | 0.440 / 0.861 / 0.51 (8233) | 0.363 / 0.813 / 0.45 (33205) | 0.280 / 0.458 / 0.61 (31621) |
| walkway | iou@0 | 0.474 / 0.710 / 0.67 (22805) | 0.449 / 0.679 / 0.66 (134409) | 0.382 / 0.514 / 0.74 (166375) |
| walkway | iou@0.5 | 0.612 / 0.894 / 0.68 (22805) | 0.559 / 0.836 / 0.67 (134409) | 0.475 / 0.634 / 0.75 (166375) |
| carpark_area | iou@0 | 0.535 / 0.759 / 0.71 (3547) | 0.491 / 0.716 / 0.69 (13381) | 0.436 / 0.582 / 0.75 (14584) |
| carpark_area | iou@0.5 | 0.688 / 0.960 / 0.72 (3547) | 0.642 / 0.933 / 0.69 (13381) | 0.579 / 0.770 / 0.75 (14584) |
| lane_divider | iou@0 | 0.244 / 0.314 / 0.78 (7798) | 0.203 / 0.308 / 0.66 (18053) | 0.179 / 0.243 / 0.74 (20034) |
| lane_divider | iou@0.5 | 0.512 / 0.815 / 0.63 (7798) | 0.459 / 0.775 / 0.59 (18053) | 0.428 / 0.574 / 0.75 (20034) |
| road_divider | iou@0 | 0.308 / 0.315 / 0.98 (2959) | 0.272 / 0.299 / 0.91 (4003) | 0.186 / 0.183 / 1.02 (4495) |
| road_divider | iou@0.5 | 0.655 / 0.794 / 0.83 (2959) | 0.583 / 0.724 / 0.81 (4003) | 0.419 / 0.415 / 1.01 (4495) |

## Table 4. Ours, divider layers from the distance field (zero crossing; GaussianCaR has no distance field)

| layer | f1@0 | f1@0.5 | f1@1 | precision@0.5 | recall@0.5 | field_mae (m) |
|---|---|---|---|---|---|---|
| lane_divider:zc | 0.137 | 0.238 | 0.280 | 0.810 | 0.140 | 0.479 |
| road_divider:zc | 0.319 | 0.554 | 0.610 | 0.820 | 0.419 | 0.553 |

Sanity (ours against the suite's stored numbers): 130 of these frames are in the stored suite evaluation `eval_curves_fusion_vel_n1000` (made on the GPU). 3992 of 4160 per-frame count arrays are identical; the others differ by at most 5 cells (a cell whose probability lies on the 0.5 threshold, CPU against GPU floats); largest per-frame score difference 0.0043, NaN pattern equal: True. From the summed counts (CPU / stored / difference): drivable_area iou@0 0.8216 / 0.8216 / +0.00002; ped_crossing iou@0 0.2696 / 0.2695 / +0.00007; walkway iou@0 0.4195 / 0.4195 / +0.00002; carpark_area iou@0 0.4798 / 0.4798 / +0.00000; lane_divider iou@0.5 0.4682 / 0.4682 / +0.00001; road_divider iou@1 0.6140 / 0.6140 / +0.00000; lane_divider:zc f1@0.5 0.2390 / 0.2391 / -0.00008; road_divider:zc f1@0.5 0.5571 / 0.5573 / -0.00025.

GaussianCaR plain IoU (thr 0.5, camera cells) by forward offset in m: drivable_area/0.0 0.856, drivable_area/0.25 0.858, drivable_area/0.5 0.858, drivable_area/0.75 0.855, drivable_area/1.0 0.855, lane_divider/0.0 0.281, lane_divider/0.25 0.281, lane_divider/0.5 0.281, lane_divider/0.75 0.272, lane_divider/1.0 0.272

Time: 2.5 s per frame on cpu with 2 threads (model 2.5 s, metrics 0.0 s; load average 3.8 at the start, 5.3 at the end); command: `temp/t16/soa_maps_v8.py runs/t16_full_s0 temp/t16/soa/soa_maps_v8_t16_full_s0.json 400 --align`.
