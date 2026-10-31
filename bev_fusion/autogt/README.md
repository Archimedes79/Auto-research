# autogt - automatic ground truth for BEV perception from nuScenes

One standalone tool: it turns the raw nuScenes data (maps, annotations, ego poses, radar) into training-ready GT per keyframe -
static BEV layers, objects, masks, and a **velocity / flow field in m/s** - and checks, shows and caches it. Nothing in it
imports `training/`; `training/gt_bev.py` is only a shim for the older code. Everything is time-based (seconds), never per frame.

```python
from autogt import AutoGT                       # run from the bev_fusion folder
gt = AutoGT("~/data/nuscenes", "v1.0-mini")     # one object (options: use_flow, use_radar, stationary_fix, map_offsets, flow_config)
g = gt.keyframe(sample_token)                   # dict of float32 layers [ix, iy] + meta
s = gt.training_sample(sample_token)            # {"prev", "cur", "next"} + the ego steps between them
gt.signature()                                  # cache key: changes with every edit of the generator or its options
```
```
python -m autogt selftest                                   # all tests (synthetic world + real keyframes + build)
python -m autogt build --version v1.0-mini --out DIR [--scenes ..] [--workers N] [--res 0.25]   # per-keyframe cache DIR/<signature>/
python -m autogt check [--holdout]                          # automatic consistency checks (+ leave-one-radar-out radar test)
python -m autogt view OUT_DIR [scene:keyframe ...] [--total]   # sample pictures (4th panel: motion relative to the static world; --total = total flow); `sheets OUT_DIR` stacks the ten defaults
python -m autogt report [--version v1.0-mini] [--holdout] [--samples N]   # ALL sequences in one pass: store the GT (data/autogt/<signature>/), run every check, write ONE PDF
                                                            # (autogt/reports/): summary, table per sequence, legend, samples drawn straight into the PDF (no picture files)
python -m autogt clean [--videos] [--pictures] [--reports] [--old-outputs] [--stale-caches DIR | --all-caches DIR] [--older-than DAYS] [--move|--delete]
                                                            # removes only what autogt generated; dry run by default, --move puts it into _to_delete/
python -m autogt video SCENE OUT.mp4 [seconds]              # driving through the GT at 12 Hz (H.264, read back)
```

## Layout
| file | content |
|---|---|
| `flow.py` | **the velocity / flow box** (dataset independent, numpy only): `estimate_flow(tracks, ego, t, grid, radar, config)` - see below |
| `nuscenes_adapter.py` | devkit records -> `Track` / `Ego` / `RadarFrame` (`NuScenesScene`); the only place that knows nuScenes' radar and annotation format |
| `nuscenes_gt.py` | `NuScenesGT`: map layers, objects, masks, velocity / flow layers, training samples, generator signature |
| `layers.py`, `grid.py` | exact supersampled rasteriser (areas, lines, footprints), occlusion mask, `GridSpec` |
| `suite.py`, `cli.py`, `__main__.py` | `AutoGT` facade; commands `selftest`, `build` (parallel, resumable cache), `check`, `view`, `sheets`, `video` |
| `checks.py`, `view.py`, `video.py`, `report.py`, `clean.py`, `encode.py`, `selftest.py` | the tools behind the commands (`encode.py`: H.264 video with a read-back check; mp4v does not play in browsers or the Claude app) |

## What a keyframe contains (ego frame at the front-camera time, x forward 0-60 m, y left +-26 m, 0.5 m or 0.25 m)
- areas `drivable_area ped_crossing walkway stop_line carpark_area` (coverage fraction, exact 4 x 4 supersampling); lines
  `lane_divider road_divider` (+ `_dist`, truncated distance in m); `centreline` + `centreline_dir`
- objects `vehicle pedestrian two_wheeler` (coverage) with `instance` (index into `meta["boxes"]`); only boxes with at least one
  lidar or radar point, the rest in `ignore`
- **velocity / flow (m/s)**: `velocity` (world velocity of the covering box in ego axes, radar-corrected), `flow_ego` (apparent
  velocity of a static world point from the ego motion, -(v + omega x r)), `flow` = sum, `flow_weight` (0 = unknown),
  `flow_sigma`; per box `vel`, `vel_source`, `vel_flag`, `vel_sigma`, `vel_known`. A displacement over any dt is `flow * dt`
- masks `camera occluded radar undefined`; `meta` with boxes, ego pose, camera, exact map vectors
- **box time = camera time** (Mirko, 2026-10-31; Waymo does the same with its `camera_synced_box`): nuScenes annotates a keyframe
  at the lidar time, the front camera took its image 36 ms earlier (median -35.9 ms, range -42.5 .. -32.6 ms), so a moving box of
  known velocity is moved by `v * (t_camera - t_annotation)` (constant velocity; size, height and yaw unchanged: median 6 cm,
  95 % 41 cm, max 88 cm on trainval). Stationary boxes (one world pose) and boxes of unknown velocity do not move. Per box in
  meta: `center`, `corners`, `world` at the camera time, `world_ann` (the annotation-time world pose), `shift` (ego axes, m),
  `shift_s` (s); `meta["annotation_timestamp"]`. `AutoGT(..., boxes_at="annotation")` / `--boxes-at annotation` gives the
  annotation positions again (nuScenes-style evaluation). The decoder needs a dt only to predict to ANOTHER time (the +-0.5 s
  keyframes: `meta["dt"]`, `meta["steps"]`); the GT of every frame is at its own camera time
- rules: stationary GT does not move (stationary objects have ONE world pose and velocity 0; map corrections only per element in
  world coordinates, `map_offsets`, never per frame)

## The flow box (`flow.py`): one function, five steps
`estimate_flow(tracks, ego, t, grid=None, radar=(), config=Config())` -> `FlowGT` (`objects` = the product: pose at `t`, velocity,
covariance, source, flag; plus the dense arrays when a grid is given)
1. ego velocity and yaw rate at `t` from the poses `t +- 0.09 s`
2. per track: stationary (all speeds < 0.5 m/s, all positions < 1 m from the median) -> one pose, v = 0; else the least-squares
   line through the observations within +-1.2 s (slope = velocity, with its covariance)
3. radar correction: detections inside the box measure the radial component of v; robust Bayesian update (3 sigma gate, capped
   weight per box)
4. flags: `radar_conflict` (most detections reject the track velocity), `radar_override` (and they agree among themselves: the
   speed along the track direction is taken from the radar), `stationary_but_moving`, `short_track`, `unobserved`
5. dense field: static world from the ego motion + each box's velocity on its footprint; weight 0 where unknown

## Validation (nuScenes mini, 404 keyframes; `python -m autogt check --holdout`)
- self-tests: synthetic world with known truth (radar lowers the velocity error, 2 Hz and 4 Hz annotations agree, a wrong direction
  of travel is found and fixed), rasteriser areas exact, static layers of the next keyframe = the current ones moved by the exact
  ego motion (IoU 0.988), a road under the ego car in all four map locations, ego flow = the exact pose change per camera frame
  (3 cm), stationary boxes fixed in the world (0.00 cm over 2394 pairs), build + reload
- map and objects: 100 % of 879 moving vehicles on the drivable area; the ego car on the road in every frame; centre-line and
  lane-divider cells on the drivable area (median 100 %); instance layer consistent (5593 of 5593)
- velocity, leave-one-radar-out (each radar scored on boxes corrected with the other four; 12,557 held-out detections in
  3,246 box-radar pairs), radial error RMSE / median / within 0.5 m/s: zero velocity 6.47 / 3.64 / 15 %, track only
  1.45 / 0.215 / 77.8 %, with radar 1.41 / 0.198 / 79.9 %; corrected better in 39.5 % of the pairs, equal 52.4 %, worse 8.2 %
  (mean RMSE change -0.036 m/s, bootstrap -0.053 .. -0.024, optimistic: pairs of one scene correlate). The annotation velocities
  agree with the radar to about 0.2 m/s; the radar adds a small gain and the flags (57 overrides, 2 conflicts over all boxes of
  mini); the remaining RMSE is outlier detections (2.6 % beyond 2 m/s: ghosts, wheels)
- the radar mask assumption: 99.9 % of all detections lie within +-60 deg of the sensor axis (measured)
- the box shift v * dt (temp/checks/radar_box_shift.py, mini, 340 moving observed vehicles >= 5 m/s, 5..45 m ahead; raw radar
  POSITIONS, which the velocity did not use; points inside the footprint + 1 m per box and radar sweep within +-0.45 s): the
  moved box keeps its 0.37 - 0.41 points at every offset, the annotation pose falls from 0.38 to 0.19 and the opposite shift to
  0.07 (|dt| 0.3 .. 0.45 s); at the real 36 ms the three variants are within noise (0.40 / 0.39 / 0.36), the check is of the
  direction and the size of the operator, not of the last centimetre. No lidar on disk, so no point-cloud check
- stationary rule of the flow box vs the devkit-speed rule of the old gt_bev: agree on 98.6 % of track-keyframes; the static layers
  of the new generator are bit-identical to the old ones, objects differ in 3 scenes by a few stationary boxes

## Known limits
- boxes exist at 2 Hz only; the yaw of a moving box is not extrapolated (a yaw rate is unknown, < 1 deg over 36 ms); a moving box
  of unknown velocity (short track, 0.6 % of the boxes) stays at the annotation time, up to v * 36 ms away from the camera view;
  a 12 Hz sweep frame (`keyframe(sd=)`) moves the keyframe's boxes by v * dt over up to +-0.25 s, a first-order extrapolation
- the offset is measured for the FRONT camera only; the other five cameras have other offsets to the annotation time
- Boston map lines are 0.3-0.7 m off the paint in places (map / localisation error); only per-element world corrections are
  allowed (`map_offsets`), and no estimator is built yet
- 25 % of the boxes stand more than 0.5 m above or below the ego ground plane (hills, overpasses): a ground-plane assumption of
  the camera lift, not a GT error
- the occlusion mask is 2D: it also hides about 12 % of boxes that annotators rate as 80-100 % visible (pedestrians behind cars)
- radar gives only the radial velocity; the tangential part comes from the annotations. Not validated against lidar scene flow
  (no lidar on disk)

## Adding a dataset
Write an adapter that builds the four inputs of `estimate_flow` (`Track`, `Ego`, `RadarFrame`, `GridSpec`) and a generator for
the static layers; `layers.py` and `flow.py` are reusable as they are.

## History
The generator was `training/gt_bev.py` (+ the first flow version `training/gt_flow.py`) and about twenty one-off scripts in
`temp/checks/` (projection checks, pitch / time-offset tests, Boston debug, map-alignment per location, viewers, videos); they
are folded into this suite and removed - git history before the commit "autogt suite" has them. What they found is in
`../plan.md` section 9.5.

## Using the velocity targets in a loss (note from the t1_r4_soa review, 2026-10-30)
About 80 % of the supervised object cells are static (velocity 0) and 20 % move: an L1 / MSE over all object cells is minimised by
predicting zero, and the gradient of a dense motion head at the BEV feature was measured ~18x below the objects group. Balance the
two groups (weight moving cells, `|velocity| > 1 m/s`, separately from static ones, or use a loss normalised per group), use
`flow_weight` (0 = unknown) as the mask and `flow_sigma` as an uncertainty (e.g. a Huber / beta-NLL scale), and report speed errors
split into moving and static cells with the zero predictor next to them. The signal is in the input: the median radar Doppler inside
a moving box reproduces the GT radial speed (corr 0.989, review notes in `temp/review/observe_2026-10-30/NOTES.md`). A `moving`
mask is not stored in the GT on purpose (it would change the generator signature and invalidate the caches); derive it from
`velocity`.
