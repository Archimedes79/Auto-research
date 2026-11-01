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
