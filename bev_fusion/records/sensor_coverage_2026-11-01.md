# Sensor coverage: one rule for training, evaluation and execution (D36, 2026-11-01)

Asked by Mirko in chat (answering D31): "If you evaluate you can only evaluate where the sensor can see something. If you train, you can only train where the sensor can see something. Maybe the system needs to know the coverage area of its sensors (radar camera) besides the grid location in the VCS and only there a fair comparison can be made." Put on the BEV list for after `t10.dual_long_s0` and its evaluation (Mirko: "Yes, put it"). The rule itself is decided as D36 (section at the end); the details are for the BEV and GT chats to settle.

## What the code does today
One real sample of the stored GT (grid 0.5 m, x 0 to 60 m, y +-26 m, 120 x 104 cells). The rig is fixed, so camera and radar masks hold for every frame. Picture: `bev_fusion/temp/sensor_coverage/coverage.png` (ignored scratch; redraw with `plot_cov.py` in the same folder, argument = one GT file of `data/cache/gt`).

| | maps (road, lines) | objects and velocity |
|---|---|---|
| training (`training/losses.py` `build_layer_targets`) | `w_map` = (1 - undefined) x 0.5 where hidden behind a box: every cell except those no sensor "sees" = 99.6 % of the grid | `w_obj` = max(camera x (1 - hidden), radar fan): 99.6 % |
| evaluation (`evaluation/bev_views.py` `masks`, `metrics.py` `in_front`) | `fov` = camera cells only (62 %); `seen` = camera or radar; `out` = outside the camera view (where a memory must help) | detections inside +-32.3 deg and x < 60 m (65.7 % of the grid); vehicle footprint on camera cells |
| execution (`training/model.py`) | camera memory writes only where `camera_visibility` (from the calibration) says the camera sees; radar memory writes everywhere (no obs_mass); no coverage channel reaches the fusion or the heads | same |

Numbers of this sample: camera 62.4 % of the grid (blind zone nearer than 6.8 m at y = 0), radar fans 99.6 %, camera-only 0 %, radar-only 37.3 %, both 62.4 %, seen by nothing 0.35 %, hidden behind boxes 5.2 % (this frame), the wedge covers 65.7 % of the grid and the camera misses 5 % of the wedge cells (the near blind zone).
Sources: `autogt/nuscenes_gt.py` (`camera` = ground cells that project into the CAM_FRONT image, `radar_coverage` = fans +-60 deg / 70 m, `occluded`, `undefined`).

## Findings
1. The radar mask is a geometric fan, not "has measurements": it covers 99.6 % of the grid and restricts nothing. 37 % of the grid is radar-only; maps are trained there but never scored, and radar cannot see road or lane layers.
2. Maps are trained on 99.6 % of the grid and scored on 62 %; objects are trained on 99.6 % and scored on 65.7 %.
3. The camera mask holds CAM_FRONT only.
4. The model gets no coverage input; only the camera memory uses a visibility mask, the radar memory none.
5. Caveat: training outside the camera view is partly intended (the memory should learn to fill unseen cells; the `out` evaluation measures it). The rule has to separate "no sensor ever sees this" from "not seen right now".

## Proposal (Mirko's direction, to be shaped by the BEV and GT chats)
- One coverage definition per output type, with two meanings: observed now, and observed earlier (memory). Maps: camera cells now, memory cells outside; objects: camera, or radar cells that really carry measurements.
- Make the radar mask mean something (cells with points in the sweep or a measured detection density), not the fan.
- Use the same masks for the training weights, the evaluation masks and the comparison of pipelines (D30: one image region; D31: wedge versus grid).
- Optional: coverage as a model input and output channel, so a planner knows what was observed.
- A change of a mask is a change of the GT: it needs a new autogt version tag (D29).

Owners: BEV system chat (training, evaluation) and GT Discussion chat (the autogt masks). The training code and the five signature files stay frozen until the BEV system chat releases them (the t10 long run is stopped at step 5030 and resumes only on Mirko's word, so there is no date).

## Decision D36 (Mirko, in chat, 2026-11-01, after this note)
"Unify the usage between training, evaluation and execution. And on the training & gt, separate what is radar data, what is camera data (lanes are camera only information for example...). Do not put a camera loss on regions, where camera cannot see. Do not put a radar loss, where radar cannot see."

First cut of what that means per output (to be refined by the BEV system and GT Discussion chats; the entries marked "to settle" need a look at the data):

| output | the information comes from | loss only where |
|---|---|---|
| lane dividers, road dividers, stop lines, crossings | camera | the camera observes the cell now |
| drivable area, walkway, carpark | camera (radar cannot see them) | the camera observes the cell now; cells seen earlier through the memory: to settle |
| camera aux head | camera | the camera observes (today `w_cam`: already so) |
| radar aux heads | radar | the radar observes, meaning cells with measurements (today `w_radar` = the fan: already separate, but too wide) |
| velocity / motion | radar Doppler, camera over time | the radar observes; the camera part: to settle |
| fused objects (vehicle, pedestrian, two-wheeler) | camera and radar | my reading, to confirm: where either sensor observes |

Today only the two aux heads follow the rule; the maps (`w_map`) and the fused objects (`w_obj`) are supervised on 99.6 % of the grid. Two more points: (a) "the camera sees the cell" exists twice, as the GT mask on the ground plane (`autogt/nuscenes_gt.py`) and as `camera_visibility` of the model (lift heights, image augmentation): they should be one function, and the difference measured first; (b) a changed loss and GT is a new regime: every arm after the change is compared with a control trained the same way, not with the stored numbers.

## Plan v6 of the GT Discussion chat and what is still open (2026-11-01, after D36)
The autogt masks start when the freeze ends (no date: the t10 long run is stopped at step 5030); a change of `nuscenes_gt.py` / `layers.py` is a new signature (GT v6) and a rebuild of trainval (5 - 8 h). Until then the chat prepares in a scratch copy. Plan: (1) two base masks, `obs_camera` (ground-plane projection of CAM_FRONT, not hidden behind a box) and `obs_radar`; per-layer masks are one documented table plus one function `autogt.coverage_for(layer)`: map layers = obs_camera, objects = obs_camera or obs_radar, velocity = obs_radar; `camera`, `radar` (fan) and `occluded` stay. (2) `obs_radar` as a static rig mask measured from the data (detection density per cell over many frames, thresholded), shown on the mini set first. (3) one function for "the camera sees the cell" shared with `training/lift_pd.camera_visibility` (the GT mask is the ground plane z = 0, the model's any height slice; the difference is measured first). Memory cells (seen earlier) are not in the GT: a model-side matter.

Mirko's two later points (chat): "there are higher performance expectations on regions where both sensors are available. Radar can just see stationary reflections and moving cars, pedestrians, etc." Reading, to be confirmed: evaluation reports per coverage class (both / camera only / radar only) with its own target; radar observes per object (a per-box radar flag from the radar points inside the box).

Open for Mirko, one at a time: (a) report every number by coverage class? (b) the radar mask definition (static density mask); (c) cells hidden behind a box count as not observed by the camera (today `w_map` halves them)? (d) fused objects: the camera supervises where it sees; where only radar observes, only objects with radar points.

Resume rule (2026-11-01, 14:45): a resume of the stopped t10 run must read the GT version recorded in its run record (v5, signature 15efbde2d1), not the newest one. The training start check of D29 therefore has to use the run's `gt_version`; the v5 stores stay on disk next to a rebuilt v6.

## Radar measured on nuScenes mini (GT Discussion chat, 404 keyframes, 2026-11-01)
Cells: one sweep of the five radars gives about 46 detections inside the grid. The share of keyframes in which a cell has a detection within 2 m is 0.11 - 0.18 in every 10 m band (max 0.35): there is no crisp "radar cannot see" region beyond the fan, so the static density mask is dropped (a threshold would be arbitrary: p >= 0.10 is 71 % of the grid, p >= 0.20 is 22 %).
Objects (7,570 annotations inside the grid), no radar point in the keyframe sweep: vehicles 34 % of the moving and 52 % of the standing ones, pedestrians 90 %, two-wheelers 81 - 84 %; median radar points per vehicle 1, per pedestrian 0. So the radar observes per object, as Mirko said.
Proposed radar part of v6 (NOT adopted, see Mirko's answer below): `obs_radar` per frame = the fan minus the footprints (+ 1 m) of boxes without a radar point in the window, plus per-box counts (`num_radar_pts`, `radar_pts_window`, `radar_observed`) in the meta. The positive radar loss covers only boxes the radar measured; cells around an unmeasured box are not scored as empty. A window of 0.5 s (6 sweeps) will raise the shares; the GT chat measures it before the freeze ends. Open question (b) is answered by data; Mirko confirms.

## Mirko's answer on the radar rule (2026-11-01)
"No because assignment of measurements to boxes is non-trivial. I want the radar to see moving objects, when they are not out of the line of sight... But thats an open topic I would say, I could think of weighting those objects down..."
Consequences: no per-box radar gate (an object is not dropped from the radar loss because no point fell into its box). The radar is expected to see moving objects that are in its line of sight, so a missing return counts as a miss of the model, not as an excuse. `obs_radar` therefore stays the fan (a radar line-of-sight version, hidden behind boxes seen from each radar, is possible later). The per-box counts (`num_radar_pts`, `radar_pts_window`) may be stored as information and for analysis. Down-weighting the objects the radar is unlikely to see (standing pedestrians, weak reflectors) instead of excluding them is an idea of Mirko's, an open topic for the BEV round, not decided. The data above stay valid: they show how large the effect is.

## Decision D37 (Mirko, in chat, 2026-11-01): fusion numbers on the fusion area only
"Fusion performance shall be reported on the fusion area only. Only that allows for apple to apple comparisons..."
The fusion area = the cells observed by camera and radar and not hidden behind a box. Today that equals the camera cells (62 % of the grid in the stored sample), because the radar fan (99.6 %) contains them. Camera-only and radar-only areas may be reported next to the headline, labelled; camera-only baselines are scored on the same fusion area. Old numbers (camera field of view) keep their label. The coverage classes derive from `obs_camera` and `obs_radar`; no extra GT layer.

## Decision D38 (Mirko, in chat, 2026-11-01): front-only rig, don't care on both sides
"I would put non visible objects to "don't care". Don't care if detected, don't care if not...." Then, on others using all cameras: "Front-only rig" (rule set for our rig, outside the fusion area everything is don't care on both sides, in training and evaluation, no 360-degree benchmark assumptions).
How others do it (searched today): KITTI ignores detections in DontCare regions; Waymo does not evaluate boxes without lidar points (Level 1 > 5 points, Level 2 1 - 5); Occ3D scores only camera-visible voxels and makes the masks optional for training; one ECCV 2024 supplement found that ignoring invisible voxels in training helped foreground classes but hurt under another metric (so: a matched control arm). They mostly score a full circle with six cameras, where almost every cell is visible; our front camera sees 62 % of the grid.
Our code today: training treats unobserved boxes as neither positive nor negative (`object_head`, `losses`), evaluation drops them from the ground truth (`gt_boxes_from_meta(observed_only=True)`) but a detection on such a box counts as a false detection: one-sided. D38 makes it two-sided and widens it from "no lidar or radar point" to "outside the fusion area or hidden". In this rig the fusion area equals the camera cells, so fused maps and objects are supervised and scored on the camera cells. D31 (wedge beyond the grid) is settled by this. Readings to confirm: per-sensor aux heads keep their own sensor area; an object counts by its centre cell; memory cells are model-side.
Rationale (Mirko, in chat, 2026-11-01): "With all cameras, NuScenes may not have to separate radar and camera visibility so much, but with jsut a front camera and a larger radar FOV, the case is different here." In the stored sample the camera sees 62 % of the grid, the radar fan 99.6 %, radar-only cells are 37 % and camera-only cells 0 %: the separation of the two visibilities is what makes the rig's rules (D36 to D38) necessary.

Aux heads (Mirko, in chat, 2026-11-01): "On aux heads yes, I am not fan of an aux head once everything trains E2E. It can be a really important tool to test, to debug a blackbox to ensure no information loss..." Consequence (D38 reading a, confirmed): the camera aux head keeps its loss on the camera area, the radar aux head on the radar area (today `w_cam_aux` = `w_radar_aux` = 0.3 in `LossConfig`); both are a test and debug tool, so they must be switchable and off in the final end-to-end arms. A proposal for the BEV chat, not decided: use them as probes on detached features (stop-gradient), so that they show whether the information survives in the black box without shaping the backbone.

## Partly visible objects: how others do it, and a proposal for D38 (b) (2026-11-01)
Mirko on the centre-cell rule: "I think that would be a primitive first step. What do others do?" Searched today (secondary sources where marked):
- KITTI: per object a truncation fraction and an occlusion level, with thresholds that make the Easy / Moderate / Hard difficulties; DontCare regions: a detection with an overlap of 0.5 or more with a DontCare area is ignored (tracking page https://cvlibs.net/datasets/kitti/old_eval_tracking.php); Complexer-YOLO (https://arxiv.org/pdf/1904.07537) reports that the BEV and 3D detection scripts do not apply it in practice, so such detections count as false.
- nuScenes: a visibility attribute per annotation, the share of the box pixels visible over the six cameras in four bins (1: 0 - 40 % up to 4: 80 - 100 %; bins from secondary sources, not checked in the devkit); users filter by level.
- Waymo: no-label-zone polygons and a per-point flag; the user supplies, per prediction, whether it overlaps an NLZ (https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md); ground-truth levels by lidar point count (Level 1 > 5 points, Level 2 1 - 5). How ground truth inside an NLZ is treated: not verified.
- Occ3D: a visibility mask per voxel (ray casting), no object rule; Argoverse 2: the rule was not found.
Pattern: nobody decides by the centre alone; the object gets a visible share (or a count) with thresholds into two or three states, and a detection is ignored by its overlap with the ignore area (0.5).
Proposal for (b), not decided: f = share of the object's footprint (rasterised on the grid) inside the fusion area and not hidden. Evaluation: care when f is high, partial in between (reported next to the headline, like Waymo Level 1 / 2), don't care when f is low; a detection with half or more of its footprint in a don't-care area is ignored. Training: a soft weight that grows with f instead of a hard 0 / 1 (this is also Mirko's earlier idea of weighting such objects down). The two thresholds come from the data (histogram of f over the objects, a sensitivity table: how many objects fall into each state), not from a guess.

Decided (Mirko, in chat, 2026-11-01): "Yes, footprint share". D38 (b) is the footprint-share rule above; the two thresholds come from the data (histogram of f over the objects and a sensitivity table).

## A soft visibility weight: max over the sensors (Mirko's question, 2026-11-01; proposal, not decided)
Mirko: does nuScenes carry radar and camera visibility values? Why not the max where both sensors are, and the one sensor's value elsewhere? Does weighting for visibility work?
- nuScenes carries per annotation: `visibility` (four bins, the share of the box pixels visible over ALL SIX cameras, so it does not describe our front camera), `num_lidar_pts` and `num_radar_pts` (points in the box, keyframe sweep). No radar visibility value beyond the count, no per-camera visibility. Our own front-camera value has to be computed (footprint share inside `obs_camera` and not hidden: D38 b); the GT chat stores the nuScenes fields and a computed `camera_observed` in v6.
- The max rule: v = max(v_camera, v_radar), a sensor that does not cover the object counting 0. It is the soft version of nuScenes' own "observed" filter (a lidar or radar point) and fits a TRAINING weight: the target is learnable when at least one sensor sees it (a smoother form: 1 - (1 - v_camera)(1 - v_radar)). It does not fit the fusion headline (D37: both sensors). It also conflicts with D38 as written (fused heads get no loss outside the fusion area; the max would give them the weight of the one sensor that sees): to be settled by experiment, not by argument.
- Catch: with `v_radar` = footprint share inside the radar fan the max is about 1 everywhere (the fan covers 99.6 % of the grid), so the weight would do nothing. `v_radar` has to carry what the radar really sees without assigning measurements to boxes: a prior per class and motion state, measured on the training set (on mini the share of objects with at least one radar point in the keyframe sweep: moving vehicles 66 %, standing vehicles 48 %, pedestrians 10 %, two-wheelers 16 - 19 %). That is Mirko's "weighting those objects down", without a per-box gate.
- Does it work? No standard recipe found. Related practice: learned uncertainty weights (heteroscedastic LiDAR detection, https://arxiv.org/pdf/1809.05590), point-count re-weighting (e.g. WeakM3D, https://arxiv.org/pdf/2203.08332), an observation-probability-weighted BCE in LiDAR reconstruction (https://arxiv.org/pdf/2302.14363); for hard ignoring the evidence is mixed (ECCV 2024 supplement above). So a matched experiment: arm A no don't-care (today), arm B hard don't-care (D38), arm C soft weight w = g(max(v_camera, v_radar prior)) with g a ramp between the two thresholds; same budget, two seeds, scored on the fusion area (D37) with the visibility levels next to it.
