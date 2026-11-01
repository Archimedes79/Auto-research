# Head design - the current state (2026-11-01)

This page is the current statement of the heads. It replaces `plan.md` sections 8.2 to 8.5 as the place to read; those sections stay as the history of how the design came about (their measurements are still valid, their design text is not always). Written from the code of this date and from the decisions D26, D27, D36, D37 and D38 (`../../bev_fusion_lane_center/experiment_execution/DECISIONS.md`); checked by a fresh critic pass the same day (22 findings, all fixed here). The picture of the whole chain is `../training/architecture.md`.

## 1. The heads as built

| head | class (file) | reads | gives | loss (`training/losses.py`) | status |
|---|---|---|---|---|---|
| time mover | `TimeMover` + `MotionHead` (`timed_head.py`) | trunk features S (after the optional 1x1 `head_channels` reduction) | S moved to t_c + dt (ego motion and the velocity field `v_hat`, bilinear splat), mass channel (0 = unknown cell), `v_hat` (2, m/s) at dt 0 | `motion_term`: L1 to the GT velocity on object cells (GT below 1 m/s set to 0), 0.1 x L1 to 0 elsewhere; trained at dt 0 only | built, shared by both heads |
| head wrapper | `TimedHead` (`timed_head.py`) | the moved state and its mass | refine block (FiLM by dt; zero-started, so an identity at initialisation for any dt), then the wrapped head | - | built |
| map head | `SemanticHead` in `TimedHead` (`bev_trunk.py`) | moved state | one channel per entry of `losses.layer_spec(centrelines, dense_objects)`: 5 areas (drivable, crossing, walkway, stop line, carpark), 2 line masks + 2 truncated distance fields (2 m), 3 dense object footprints; with `centrelines` also a centre-line mask and (cos, sin) | areas: BCE + 0.5 x soft IoU; lines: dilated Dice + Huber on the distance field; dense objects: BCE + IoU | built |
| object head | `ObjectHead` in `TimedHead` (`object_head.py`) | moved state | CenterPoint maps for vehicle / two-wheeler / pedestrian: heatmap, offset, log size, (sin, cos), log radius, velocity; BatchNorm in the shared layer (train / eval differ) | focal loss on the heatmap, L1 at the GT centre cells (`head_loss`), at most `max_obj` = 128 objects per frame | built |
| direction option | `ObjectHead(direction=True)` (`model.object_direction`) | - | `dir` logit: the heading is a sign-free axis in [-45, 135) deg plus a front / back class (SECOND); the axis loss is `min(L1(r, t), L1(r, -t))`, the class a BCE (weight 0.2) | `head_loss` | built; in the incumbent |
| hurdle option | `ObjectHead(moving=True)` (`model.object_moving`, t11, commit 6fe7b57) | - | `mov` logit (moving = speed > 0.5 m/s); the velocity is regressed on moving centres only and decoded as 0 where `mov` < 0 | `head_loss` (BCE weight `w_obj_mov` 1.0; velocity weight `w_obj_vel` = 1.0 in every t10 / t11 config, default 0.2) | built, an **option**: not promoted (`records/champion.json`: `options` empty) |
| refiner option | `BoxRefiner` (`model.refine`) | object head features at the target time | four deformable samples per candidate, residual box + quality | `refine_loss`, weight `w_refine` | built; ran in `t1_r3_combined`, **off** in every config since (t2 - t14) |
| aux heads | `SemanticHead` on the branch BEVs (`model.aux_heads`) | camera BEV / radar BEV | camera aux: maps and objects on the camera cells that are not occluded; radar aux: objects and the `radar_stationary` proxy channel on the radar cells, no maps | fixed weights 0.3 (`w_cam_aux`, `w_radar_aux`) | built, training only, switchable (D38 (a): a test and debug tool, off in the final E2E arms) |
| dense output | `extract.dense_boxes`, `extract.soft_line_map` (`model.dense_outputs`) | the head maps | per cell a peak weight + box + heading vector (the axis x tanh of the direction logit) and the velocity x sigmoid of the moving logit; a soft line map | none (nothing trains on them yet) | built, off in training and timing; on for the planner and the prototype |
| selection | `extract.select_boxes` | dense boxes | the box dicts of `object_head.decode` | - | built; the post-processing of `model.boxes()`. The evaluation adapter still scores with `object_head.decode` (gap 4), and `refine_boxes` is a further box step when the refiner is on; the line cells stay `evaluation/metrics.line_cells` |

Weighting between the four task groups (areas, lines, objects, motion): learned Kendall weights, clamp [-3, 1] (D27); time weights w(dt) 1 / 0.5 / 0.25 with dt = 0 only for the first 20 % of the steps (D27). The object groups are `losses.NUSC_GROUPS` (vehicle, two_wheeler, pedestrian), not the synthetic classes of `object_head.GROUPS`.

**Other loss terms and arms that exist** (all off or fixed-weight in the incumbent): the `radar_stationary` proxy target (a hook, skipped when the GT has no such layer) and its aux loss; the radar slot encoder's reconstruction side loss (`radar_kind` `delta_slots`, `w_radar_slot` 0.1); the splat-correction penalty `w_correction` (the mover has no correction in the model, tests only); the consistency terms (`LossConfig.consistency`: object velocity against `v_hat`; `TrainConfig.consistency`: cross-time); `area_overlap` "semimetric", `line_boundary`; `TrainConfig.balancing` fixed / uwso.

## 2. What the loss supervises today (the masks)

All masks come from layers of the GT dict (`autogt`): `undefined` (neither sensor), `occluded` (behind a GT box seen from the camera), `camera`, `radar`, `ignore` (a box without any lidar or radar point).

| weight | map areas and lines (`w_map`, `w_lines`) | objects and velocity (`w_obj`) | camera aux (`w_cam`) | radar aux (`w_radar`) |
|---|---|---|---|---|
| rule today | 0 on `undefined`, 0.5 on `occluded`, x registration weight on lines | max(camera and not occluded, radar), x 0 on `ignore`, x 0.5 on interpolated labels | camera and not occluded | radar |
| with `fusion_area_only` (D38 (b) first step) | x fusion area (camera x radar) | x fusion area | unchanged (own sensor area, D36) | unchanged |

On 40 real frames the option lowers the mean map weight 0.878 -> 0.546 and drops 46.6 % of the supervised object centres (369 -> 197): the objects beside the car, outside the camera's view, are supervised by the radar alone today (`../experiments/t14_dontcare/results/fusion_area_real_gt.txt`, `tests/test_fusion_area_real.py`). The option does not reach the refiner's box mask (`losses.refine_gt_boxes`), so `startable.py` refuses it together with `model.refine`.

In every row a cell that is "unknown at the target time" (splat mass < 0.5, dt != 0) counts 0.5. An object whose centre cell has weight 0 or whose box is unobserved is **ignored**: its heatmap peak region gets a constant logit, so it is neither a positive nor a negative.

## 3. What changed against plan section 8

| plan text | now |
|---|---|
| 8.3 stage-1 head: centre offset, size, yaw (sin, cos), velocity | plus the direction class (sign-free axis) and the hurdle logit; decode zeroes the velocity of static centres |
| 8.3 S5b: the default is box propagation with the bounded correction (A); the FiLM head on the splatted state stays as the alternative arm | the built chain uses the alternative: the object head is a `TimedHead` on the moved state; `propagate_boxes` / `BoxCorrection` / `TimeObjectHead` remain in `object_head.py` as the box-propagation control, not on the chain |
| 8.4 masks: occluded 0.5, unobserved boxes ignored | still true as the default; D36 / D37 / D38 add the fusion-area rule (option in section 2) |
| 8.4 "decisions still needed" (5), (6) | decided in D27: w(dt) as above, Kendall weights |
| 8.5 `CentreHead` (distance field, 8 direction bins, split / merge points), decided dense-first in D26 | **not built**; `centrelines=True` is still the old design (mask + L1 on (cos, sin) inside the map head). See section 5, gap 1 |
| 8.3 velocity regression at the centre cells (L1) | built, but at effective batch 3 the **object head's `vel` branch** was dead in 28 of 30 runs (zero is the L1 median of a zero-inflated target); the hurdle revives it. The time mover's own velocity field `v_hat` (8.2) is a separate quantity |

## 4. Decisions that touch the heads, and where they stand

| decision | what it asks of the heads | in code |
|---|---|---|
| D26 | lane centre-lines: a dense head trained with the others, a rule graph at inference | not built (section 5, gap 1) |
| D27 | Kendall weights, w(dt) 1 / 0.5 / 0.25 | built |
| D36 | camera loss only where the camera sees, radar loss only where the radar sees; one coverage function for training, evaluation and execution | aux heads: built (`w_cam`, `w_radar`); the fused heads: see D38; `precompute.py` makes `vis_cam` (a placeholder `vis_radar`) for the memory write masks, which is a third definition next to the GT layers (gap 6) |
| D37 | fusion numbers on the fusion area only | evaluation side (the Evaluation chat); the training side is the option below |
| D38 (a) | aux heads keep their own sensor area, switchable | built (`aux_heads`) |
| D38 (b) | an object counts by the share f of its footprint that is observed: care from 0.5, partial 0.1 to 0.5, don't care below 0.1; occlusion softened | **first step built as an option**: `LossConfig.fusion_area_only` (default off) = hard don't-care outside the fusion area (camera x radar) for the fused heads. The share per box exists since autogt v7 (`meta["boxes"][i]["share_fusion"]`, v8 is the same GT made fast), the softer occlusion with it; the trainval GT of v7 / v8 is not built, so the soft weight is not in the loss yet |

The order Mirko confirmed: evaluation states first (care / partial / don't care in the scoring, no retraining); the training masks as an option with a matched control (arm A today's loss, arm B the option, same budget, two seeds, both scored on the fusion area). The control is planned: `../experiments/t14_dontcare/experiment.md` (arms `training/configs/t14_dc_ctl_s0.json` and `t14_dc_fusion_s0.json`: the hurdle model continued 1000 steps with and without the option; prediction, rule and the start gates are written there). **No run until Mirko lifts the training stop of 2026-11-01.**

## 5. Open gaps (named, with what blocks them)

| # | gap | blocked by | next step |
|---|---|---|---|
| 1 | The centre-line head of plan 8.5 (D26) is not built; the Spurmitte output of this system is the old mask + (cos, sin) design | needs GT additions first: autogt has the centre-line mask and direction, not the distance field, the direction bins or the split / merge points from the map links; and an assembly + oracle for the metric | S1 of plan 8.5: GT additions + oracle + metric on CPU, then the head |
| 2 | The footprint-share weight (D38 (b)) and the softened occlusion in the loss | the trainval GT of autogt v7 / v8 (the build was stopped at 1,455 keyframes; v8 = v7 made fast); no training while the stop holds | read `share_fusion` in `build_layer_targets` as a soft weight (0 below 0.1, 1 from 0.5), tested on a mini GT |
| 3 | The refiner matches GT inside the network (`model._refine_train`); it belongs in the loss | `model.py` is edited by several chats; harmless while `refine` is off | move the matching into `losses.py` when `model.py` is quiet |
| 4 | Two box extractors: the evaluation adapter scores with `object_head.decode`, `model.boxes()` with `extract.select_boxes` (equal box for box, tested) | `evaluation/bev_adapter.py` belongs to the Evaluation chat | let the adapter read `select_boxes` |
| 5 | Hurdle costs: front mAP2 -0.014 and vehicle AP@0.5 0.254 -> 0.207, static error 0.03 -> 0.28 m/s; pedestrian velocity skill 0.014 | one seed each; the long run `t10_dual_long_s0` was stopped at step 5030 and Mirko answered "no" to finishing it | the seed twin of B2 (about 85 min); the dense motion field (`experiments/t12_motion_block`) |
| 6 | The fusion area has three definitions: the loss reads the GT layers `camera` x `radar` (`losses.py`), `precompute.vis_fusion` is a second (any-height camera visibility, used by the memories), the evaluation masks a third; the GT `camera` is a ground-plane projection, the model's `vis_cam` is any-height. `vis_radar` is a placeholder (every cell; the fans cover 99.6 % of the grid), so the fusion area equals the camera cells | `autogt.coverage` (v7) defines the shared version; unifying touches `precompute.py`, `losses.py` and the evaluation | one function for "the camera sees the cell" (the coverage note) |
| 7 | The GT of a run is not pinned by its code snapshot: a start reads the live autogt (v8 today, the base was trained on v5, signature 15efbde2d1) | a pinned copy at `<out_root>/<name>/autogt` is the way (as the t10 long run); `startable.py` now checks it | pin v5 for the t14 arms (card gate G2) |

Numbers in this page come from `../records/morning_2026-11-01.md`, `../experiments/t11_vel_hurdle/experiment.md` (cold VAL, 1000 frames, 82 scenes), `temp/box_vel/out/alive_table.txt` (28 dead of 30 runs; the t11 card says 24, stale) and the code named in each row.
