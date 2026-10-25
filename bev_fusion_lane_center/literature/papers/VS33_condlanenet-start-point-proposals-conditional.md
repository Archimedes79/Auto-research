# VS33 - CondLaneNet: start-point proposals, conditional convolution, row-wise shape and per-row range

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S1, S2, S3.*

## Papers

CondLaneNet (Liu et al., ICCV 2021) https://arxiv.org/abs/2105.05003

## Idea

Each lane instance is detected at its START point (CenterNet-style heat map); a dynamic kernel generated at that point convolves the feature map into that lane's row-wise shape; the vertical range is a per-row binary "does the lane pass this row"; a recurrent module lets forks share one start point.

## Formulation

proposal heat map 1 x Hp x Wp + parameter map Cp x Hp x Wp (sizes not stated); x per row = expectation of a softmax over X grid cells (loc = sum_j j p_ij) plus an offset map; range: per-row logistic output, lane spans v_min..v_max; RIM = LSTM emitting at each step a 2-way state (continue / stop) and a kernel vector, so one proposal yields several lanes until "stop"; loss = L_point (focal) + 1.0 L_row (L1 on expected x) + 1.0 L_range (BCE per row) + 0.4 L_offset + 1.0 L_state (CE); RIM only on CurveLanes.

## Input

image_feat. output: polyline lane instances (start point, rows, range).

## Training

dense, no Hungarian.

## Differentiable?

yes except peak picking.

## Sensors and data

images (CULane, TuSimple, CurveLanes).

## Reported results

F1 CULane 78.14 / 78.74 / 79.48 (small / medium / large; 220 / 152 / 58 FPS), TuSimple 97.01 / 96.98 / 97.24, CurveLanes 85.09 / 85.92 / 86.10 (RTX 2080).

## Compute

FPS above.

## Code

URL from memory, not opened: https://github.com/aliyun/conditional-lane-detection

## Fit to our project

reuse_idea - S - (a) lane END as a per-row existence classification (not a survival product); (b) continue / stop per RIM step is a discrete halting decision for forks (compare E94 events).

## Needs

row-wise parametrisation.

## Conflicts

one x per row.

## Tried in our experiments

E94 events continue / end / fork / merge.
