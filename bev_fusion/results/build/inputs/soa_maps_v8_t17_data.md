# Map layers: runs/t17_data (ours) against GaussianCaR, GT v8 (0.5_4_6a7aa7864a), VAL

384 val frames of 41 scenes (the 384 of the 400 frames of the old scoring that are samples of this val set (16 are the first / last keyframe of a scene)), camera cells only (the GT camera mask, as `seg_fov`), checkpoint ckpt_00006000.pt step 6000 (EMA), memory empty before every frame, dt 0. VAL, not TEST. The denominator of each number is the number of GT cells (`n_gt`, summed over the frames) in the table. GaussianCaR: 6 cameras + radar, 360 degrees, trained for the nuScenes maps (its cached logits, forward offset 0.5 m); ours: front camera + radar. Prior: per-cell GT mean of 500 train frames (null). Tolerance t: a cell counts as right if the other map has a cell within t metres (iou@0 = plain IoU).

## Table 1. Prediction threshold 0.5, all ranges (the numbers to quote)

| layer | metric | n_gt cells | prior | ours | GaussianCaR | ours / GaussianCaR | ours (record, 1000 frames, 82 scenes) |
|---|---|---|---|---|---|---|---|
| drivable_area | iou@0 | 1311823 | 0.433 | 0.818 | 0.858 | 0.95 | 0.767 |
| drivable_area | iou@0.5 | 1311823 | 0.473 | 0.873 | 0.912 | 0.96 | 0.816 |
| drivable_area | iou@1 | 1311823 | 0.510 | 0.903 | 0.936 | 0.97 | 0.847 |
| ped_crossing | iou@0 | 73059 | 0.000 | 0.272 | 0.528 | 0.51 | 0.257 |
| ped_crossing | iou@0.5 | 73059 | 0.000 | 0.345 | 0.650 | 0.53 | 0.324 |
| ped_crossing | iou@1 | 73059 | 0.000 | 0.412 | 0.720 | 0.57 | 0.385 |
| walkway | iou@0 | 323589 | 0.000 | 0.409 | 0.594 | 0.69 | 0.402 |
| walkway | iou@0.5 | 323589 | 0.000 | 0.511 | 0.733 | 0.70 | 0.492 |
| walkway | iou@1 | 323589 | 0.000 | 0.585 | 0.799 | 0.73 | 0.557 |
| carpark_area | iou@0 | 31512 | 0.000 | 0.487 | 0.656 | 0.74 | 0.239 |
| carpark_area | iou@0.5 | 31512 | 0.000 | 0.636 | 0.857 | 0.74 | 0.296 |
| carpark_area | iou@1 | 31512 | 0.000 | 0.706 | 0.916 | 0.77 | 0.324 |
| lane_divider | iou@0 | 45885 | 0.000 | 0.203 | 0.281 | 0.72 | 0.191 |
| lane_divider | iou@0.5 | 45885 | 0.000 | 0.457 | 0.690 | 0.66 | 0.430 |
| lane_divider | iou@1 | 45885 | 0.000 | 0.583 | 0.833 | 0.70 | 0.546 |
| road_divider | iou@0 | 11457 | 0.000 | 0.257 | 0.258 | 0.99 | 0.239 |
| road_divider | iou@0.5 | 11457 | 0.000 | 0.542 | 0.614 | 0.88 | 0.494 |
| road_divider | iou@1 | 11457 | 0.000 | 0.614 | 0.756 | 0.81 | 0.560 |

## Table 2. Best of the thresholds {0.3, 0.5, 0.7} per layer and metric, chosen on these frames (IN-SAMPLE, for the model trained with another loss)

| layer | metric | n_gt cells | prior | ours | GaussianCaR | ours / GaussianCaR |
|---|---|---|---|---|---|---|
| drivable_area | iou@0 | 1311823 | 0.505 | 0.818 (thr 0.5) | 0.858 (thr 0.5) | 0.95 |
| drivable_area | iou@0.5 | 1311823 | 0.543 | 0.873 (thr 0.5) | 0.912 (thr 0.5) | 0.96 |
| drivable_area | iou@1 | 1311823 | 0.578 | 0.903 (thr 0.5) | 0.936 (thr 0.5) | 0.97 |
| ped_crossing | iou@0 | 73059 | 0.000 | 0.274 (thr 0.3) | 0.528 (thr 0.5) | 0.52 |
| ped_crossing | iou@0.5 | 73059 | 0.000 | 0.345 (thr 0.5) | 0.650 (thr 0.5) | 0.53 |
| ped_crossing | iou@1 | 73059 | 0.000 | 0.412 (thr 0.5) | 0.720 (thr 0.5) | 0.57 |
| walkway | iou@0 | 323589 | 0.000 | 0.420 (thr 0.3) | 0.594 (thr 0.5) | 0.71 |
| walkway | iou@0.5 | 323589 | 0.000 | 0.526 (thr 0.3) | 0.733 (thr 0.5) | 0.72 |
| walkway | iou@1 | 323589 | 0.000 | 0.611 (thr 0.3) | 0.799 (thr 0.5) | 0.76 |
| carpark_area | iou@0 | 31512 | 0.000 | 0.490 (thr 0.7) | 0.671 (thr 0.7) | 0.73 |
| carpark_area | iou@0.5 | 31512 | 0.000 | 0.648 (thr 0.7) | 0.857 (thr 0.5) | 0.76 |
| carpark_area | iou@1 | 31512 | 0.000 | 0.715 (thr 0.7) | 0.916 (thr 0.5) | 0.78 |
| lane_divider | iou@0 | 45885 | 0.000 | 0.205 (thr 0.7) | 0.367 (thr 0.7) | 0.56 |
| lane_divider | iou@0.5 | 45885 | 0.000 | 0.461 (thr 0.7) | 0.763 (thr 0.7) | 0.60 |
| lane_divider | iou@1 | 45885 | 0.000 | 0.584 (thr 0.7) | 0.833 (thr 0.5) | 0.70 |
| road_divider | iou@0 | 11457 | 0.000 | 0.261 (thr 0.7) | 0.334 (thr 0.7) | 0.78 |
| road_divider | iou@0.5 | 11457 | 0.000 | 0.547 (thr 0.7) | 0.683 (thr 0.7) | 0.80 |
| road_divider | iou@1 | 11457 | 0.000 | 0.615 (thr 0.7) | 0.756 (thr 0.5) | 0.81 |

## Table 3. By range, threshold 0.5 (ours / GaussianCaR / ratio; n_gt of the bin in brackets)

| layer | metric | 0-20 m | 20-40 m | 40-60 m |
|---|---|---|---|---|
| drivable_area | iou@0 | 0.918 / 0.940 / 0.98 (219414) | 0.838 / 0.898 / 0.93 (528211) | 0.763 / 0.793 / 0.96 (564198) |
| drivable_area | iou@0.5 | 0.965 / 0.986 / 0.98 (219414) | 0.893 / 0.952 / 0.94 (528211) | 0.821 / 0.850 / 0.97 (564198) |
| ped_crossing | iou@0 | 0.350 / 0.704 / 0.50 (8233) | 0.285 / 0.648 / 0.44 (33205) | 0.236 / 0.378 / 0.62 (31621) |
| ped_crossing | iou@0.5 | 0.447 / 0.861 / 0.52 (8233) | 0.362 / 0.813 / 0.44 (33205) | 0.300 / 0.458 / 0.65 (31621) |
| walkway | iou@0 | 0.462 / 0.710 / 0.65 (22805) | 0.439 / 0.679 / 0.65 (134409) | 0.376 / 0.514 / 0.73 (166375) |
| walkway | iou@0.5 | 0.604 / 0.894 / 0.68 (22805) | 0.547 / 0.836 / 0.65 (134409) | 0.467 / 0.634 / 0.74 (166375) |
| carpark_area | iou@0 | 0.523 / 0.759 / 0.69 (3547) | 0.483 / 0.716 / 0.67 (13381) | 0.481 / 0.582 / 0.83 (14584) |
| carpark_area | iou@0.5 | 0.671 / 0.960 / 0.70 (3547) | 0.633 / 0.933 / 0.68 (13381) | 0.631 / 0.770 / 0.82 (14584) |
| lane_divider | iou@0 | 0.221 / 0.314 / 0.70 (7798) | 0.214 / 0.308 / 0.69 (18053) | 0.185 / 0.243 / 0.76 (20034) |
| lane_divider | iou@0.5 | 0.484 / 0.815 / 0.59 (7798) | 0.477 / 0.775 / 0.62 (18053) | 0.428 / 0.574 / 0.75 (20034) |
| road_divider | iou@0 | 0.295 / 0.315 / 0.93 (2959) | 0.283 / 0.299 / 0.95 (4003) | 0.198 / 0.183 / 1.09 (4495) |
| road_divider | iou@0.5 | 0.599 / 0.794 / 0.75 (2959) | 0.594 / 0.724 / 0.82 (4003) | 0.444 / 0.415 / 1.07 (4495) |

## Table 4. Ours, divider layers from the distance field (zero crossing; GaussianCaR has no distance field)

| layer | f1@0 | f1@0.5 | f1@1 | precision@0.5 | recall@0.5 | field_mae (m) |
|---|---|---|---|---|---|---|
| lane_divider:zc | 0.181 | 0.286 | 0.320 | 0.812 | 0.173 | 0.477 |
| road_divider:zc | 0.369 | 0.579 | 0.618 | 0.828 | 0.446 | 0.531 |

Sanity (ours against the suite's stored numbers): 130 of these frames are in the stored suite evaluation `eval_curves_fusion_vel_n1000` (made on the GPU). 3992 of 4160 per-frame count arrays are identical; the others differ by at most 3 cells (a cell whose probability lies on the 0.5 threshold, CPU against GPU floats); largest per-frame score difference 0.0022, NaN pattern equal: True. From the summed counts (CPU / stored / difference): drivable_area iou@0 0.8214 / 0.8214 / -0.00000; ped_crossing iou@0 0.2644 / 0.2643 / +0.00004; walkway iou@0 0.4120 / 0.4120 / -0.00002; carpark_area iou@0 0.5079 / 0.5080 / -0.00009; lane_divider iou@0.5 0.4578 / 0.4578 / -0.00002; road_divider iou@1 0.5983 / 0.5983 / +0.00003; lane_divider:zc f1@0.5 0.2720 / 0.2720 / +0.00003; road_divider:zc f1@0.5 0.5771 / 0.5772 / -0.00010.

GaussianCaR plain IoU (thr 0.5, camera cells) by forward offset in m: drivable_area/0.0 0.856, drivable_area/0.25 0.858, drivable_area/0.5 0.858, drivable_area/0.75 0.855, drivable_area/1.0 0.855, lane_divider/0.0 0.281, lane_divider/0.25 0.281, lane_divider/0.5 0.281, lane_divider/0.75 0.272, lane_divider/1.0 0.272

Time: 2.5 s per frame on cpu with 2 threads (model 2.5 s, metrics 0.0 s; load average 4.9 at the start, 6.1 at the end); command: `temp/t16/soa_maps_v8.py runs/t17_data temp/t16/soa/soa_maps_v8_t17_data.json 400 --align`.
