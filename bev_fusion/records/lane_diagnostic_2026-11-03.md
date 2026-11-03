# Lane diagnostic of t16.full_s0 (step 1 of plan_2026-11-03.md, 2026-11-03)

What it answers (the rule was written in `../plan/plan_2026-11-03.md` section 2 before the numbers were read): is the lane weakness a resolution cap of the 0.5 m grid, a range problem, or a problem of the near field too? No run: the numbers are read from the stored evaluation of `t16.full_s0` (`runs/t16_full_s0/eval/metrics.json` = VAL, 1000 frames of 82 scenes; `metrics_train.json` = TRAIN split, frame count not re-checked today), key `seg_fov/seg/dt+0.00/lane_divider/all/<range>/iou@<tol>`. A diagnostic: it feeds no verdict.

| lane divider IoU, front field of view, dt 0 | 0-20 m | 20-40 m | 40-60 m | all |
|---|---|---|---|---|
| VAL, tolerance 0.5 m | 0.502 | 0.430 | 0.399 | 0.429 |
| TRAIN, tolerance 0.5 m | 0.623 | 0.564 | 0.518 | 0.554 |
| VAL, tolerance 1 m | 0.595 | 0.553 | 0.527 | 0.549 |
| TRAIN, tolerance 1 m | 0.721 | 0.672 | 0.635 | 0.665 |

Reading (by hand: the rule of the plan, applied to these numbers):

1. **The 0.5 m grid is not the cap.** The oracle row of `evaluation/bev_gtcheck.py` (the GT used as the prediction; its selftest asserts IoU 1 at the metric's tolerance, a shift of k cells recovers at tolerance >= k cells) says the grid and the metric can carry 1.0. The rule's "ceiling below 0.8" case does not apply. What does cap the number is the map: the nuScenes map sits 0.3 - 1 m beside the paint in places (`head_design.md` section 6 item 4), which is 1 - 2 cells, so IoU@0.5 cannot reach 1 even for a perfect network; IoU@1 is the fairer line number.
2. **The near field is poor too.** 0.50 (VAL) at 0-20 m, where one stride-16 image pixel is about 0.3 - 0.6 m and the camera sees the paint best. The fall with range is only 0.10 (0.50 -> 0.40). The rule's second case applies: L4 (the stride-4 branch, plan section 2) is the lever, on in the bundle.
3. **A train / val gap of about 0.12 at every range** (0.62 against 0.50 near, 0.52 against 0.40 far) is as large as the whole range fall. So part of the weakness is generalisation, not features: L4 raises the features, it does not close that gap. Not addressed in t20 (the split is the large one already: 428 train scenes); it is a named weak spot for the report.
4. **Not measured here:** lane IoU by range for the road divider and the centre-lines (the centre-line is not trained in t16.full); the share of the VAL error that is map offset (needs a registration estimate, `head_design.md` gap 9).

Consequence for t20: L4 on (extended to `camera_features` 'dual'), the 1 m centre-line head, no other change of the lift. The comparison rows of the card: these numbers (VAL, 1000 frames).
