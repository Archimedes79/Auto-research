# autogt - guide for agents (read this first, README.md is the reference)

autogt turns raw nuScenes (map, 3D box annotations, ego poses, radar) into **training-ready BEV ground truth per keyframe**:
static map layers, object footprints, masks, and a velocity / flow field in m/s. Standalone: nothing in it imports `training/`.
Everything is in seconds (never "per frame"). Unit of work = one keyframe (sample token, 2 Hz, 39-41 per sequence; nuScenes mini = 10
sequences / 404 keyframes, trainval = 850 / 34,149).

## 1. Which entry point? (the whole interface)
| I want | do this |
|---|---|
| the GT of keyframes that are already generated (training, analysis) | `autogt.load(token)` - 0.1 s, no nuScenes needed; the tokens of a sequence: `autogt.scene_tokens("scene-0561", "v1.0-trainval")` (json tables, 1 s) |
| to generate the GT of a nuScenes folder (all sequences or some) and store it where `load` reads it | `autogt.generate("~/data/nuscenes_full")` (or `python -m autogt generate ~/data/nuscenes_full [--out DIR] [--scenes scene-0061 ...] [--stride N]`): finds the version in the folder, memory-light, resumable (a rerun writes only what is missing, a finished group loads no devkit), returns `{"out", "signature", "gt_version", "version", "scenes", "keyframes", "written", "release"}`; trainval 5-8 h; default `out` = `data/autogt` -> `<signature>/<token>.npz` + `index.json`; one generate / report per folder at a time |
| a small or a full run | `autogt.generate(dataroot, set_name="small")` (the 10 sequences of mini, ~25 min, small experiments) or `"full"` (all 850 of trainval, 5-8 h, releases); shell `--set small|full`; `python -m autogt sets` shows how much of each is stored; the manifest in `index.json` (`gt_version`, `commit`, `dirty`, `sets_complete`, `release`) says what a folder is: `release: true` = complete, released GT version, clean tree, `full` held |
| to know that the raw nuScenes files are all on disk before a long run | `python -m autogt data '~/data/nuscenes_full' [--set small|full]` (exit 1 and the missing files if not; trainval takes minutes); fetching / verifying / linking the raw data is `../datasets/NuScenes/` (downloads only with the user's OK) |
| to generate single keyframes in code (other options, a quick look) | `AutoGT(dataroot, version).keyframe(token)` |
| prev / cur / next keyframe + the ego motion between them | `AutoGT(...).training_sample(token)` (section 5) |
| GT of ALL sequences stored + every check + one PDF | `python -m autogt report --version v1.0-mini` (trainval: `--version v1.0-trainval`, 5-8 h, resumable) |
| to look at a keyframe / drive through a sequence | `python -m autogt view OUT_DIR <token>` (any STORED keyframe, mini or trainval) or `... view OUT_DIR scene-0061:8` (mini, generated now) / `python -m autogt video scene-0061 out.mp4 [seconds]` (mini; a trainval sequence: `python -m autogt video scene-0441 out.mp4 20 --version v1.0-trainval`) |
| to be sure after editing the generator | `python -m autogt selftest` (about 5 min; also checks that this file lists every key) |
| to remove what autogt generated | `python -m autogt clean --stale-caches data/autogt` (dry run; `--move` / `--delete` to act; the caches of released GT versions are left alone, `--include-released` lists them and they can only be moved) |

Python (all of it): `from autogt import generate, AutoGT, load, scene_tokens, current_signature, GridSpec`.
```python
import autogt
toks = autogt.scene_tokens("scene-0561", "v1.0-trainval")      # 40 sample tokens in time order (json tables only; mini: default version)
g = autogt.load(toks[8])                                        # dict: 24 float32 layers + `instance` (int32) + "meta" (tables below); raises FileNotFoundError
                                                                # (naming the folders that exist) if the token is not stored
gen = autogt.AutoGT("~/data/nuscenes", "v1.0-mini")             # generate: mini loads in 4 s, the whole devkit is fine
g = gen.keyframe(gen.keyframes("scene-0061")[8])                # same dict, generated now (2-3 s)
# trainval: `AutoGT(..., light=True, scenes=[...])` still reads the big json tables (minutes on a busy box): prefer `load`
gen.signature() == autogt.current_signature()                   # the cache key (section 4)
```
A cache folder without `index.json`, or with `"complete": false` in it, belongs to a `report` / `build` that is still running or was
interrupted (`build` writes a provisional index with the resolution and supersampling before the first keyframe and the final one at the
end; a resume refuses another resolution): its files are valid, but the sequence list is incomplete. `clean --stale-caches` leaves a
provisional folder alone for an hour after its last write. CLI details: `python -m autogt --help`; the commands you need most are `report`,
`view`, `selftest`.

## 2. Run it (this box is shared)
All commands run **from the repository root (`C:\Auto-research`) inside WSL** with the conda env `gcar`; from Windows prefix `wsl`:
`wsl bash autogt/run_nice.sh -m autogt selftest` (nice 15, one thread; runs any script the same way:
`... run_nice.sh my_script.py`; a running `report` is stopped with `wsl bash -c "pkill -f '[a]utogt report'"`). Rules: one process, no worker pools while a training round runs (`uptime`, `free -m` first),
the full trainval devkit needs 8+ GB (use `light=True, scenes=[...]`; `report` does it automatically), run long jobs in the
background and wait with a bounded loop. Data: `~/data/nuscenes` (mini, 10 sequences, 404 keyframes), `~/data/nuscenes_full`
(trainval, 850 sequences, 34,149 keyframes). Code and output are apart: the code is the package `autogt/` at the repository root (in git); everything
autogt generates is under `data/autogt/` (not in git): stored GT `<signature>/<token>.npz` + `index.json`, reports (PDF)
`reports/`, example videos / pictures `examples/`. Stored GT is derived data and can always be rebuilt (5-8 h for trainval); the raw
nuScenes data is on D: (`~/data/nuscenes*` are links to it) and is never written to.

## 3. What a keyframe contains
Grid: ego frame at the **front-camera time**, x forward 0..60 m, y left -26..26 m, 0.5 m cells (`GridSpec(res=0.25)` for 0.25 m)
-> arrays `[ix, iy]` of shape (120, 104); `iy = 0` is the RIGHT edge (y = -26), cell centre x = (ix + 0.5) * res, y = -26 + (iy + 0.5) * res.
Vectors are (2, nx, ny) = (x, y) components in ego axes. All layers are float32 (stored as float16), `instance` int32.

| key | shape | meaning |
|---|---|---|
| `drivable_area` `ped_crossing` `walkway` `stop_line` `carpark_area` | (nx, ny) | map areas, coverage fraction 0..1 (exact 4x4 supersampling) |
| `lane_divider` `road_divider` | (nx, ny) | 1-cell line masks |
| `lane_divider_dist` `road_divider_dist` | (nx, ny) | distance to the nearest line in m, truncated at 2 |
| `centreline` | (nx, ny) | mask of the lane / lane-connector centre-lines |
| `centreline_dir` | (2, nx, ny) | direction of travel (cos, sin of the heading) on `centreline` cells |
| `vehicle` `pedestrian` `two_wheeler` | (nx, ny) | footprint coverage of **observed** boxes (at least one lidar or radar point) |
| `instance` | (nx, ny) int32 | index into `meta["boxes"]` of the box that covers the cell, -1 = none |
| `ignore` | (nx, ny) | boxes WITHOUT any lidar / radar point: neither object nor free space - leave out of every loss |
| `velocity` | (2, nx, ny) | world velocity (m/s, ego axes) of the covering box, radar-corrected; 0 elsewhere. Static world = 0 |
| `flow_ego` | (2, nx, ny) | apparent velocity of a static world point caused by the ego motion, -(v + omega x r) |
| `flow` | (2, nx, ny) | `flow_ego + velocity`: the apparent motion in the camera frame; displacement over any dt = `flow * dt` |
| `flow_weight` | (nx, ny) | 1 = velocity known, 0 = unknown (ignored boxes, unknown velocity): the supervision mask |
| `flow_sigma` | (nx, ny) | std of the velocity estimate in m/s |
| `camera` | (nx, ny) | ground cells that project into the CAM_FRONT image |
| `occluded` | (nx, ny) | ground cells a box hides from the camera, by the box's real height (v7: no longer an infinitely high box; section 3a) |
| `radar` | (nx, ny) | cells inside the +-60 deg fan of one of the five radars |
| `undefined` | (nx, ny) | cells neither camera nor radar can see |

`meta` (dict): `sample` / `scene` (tokens), `timestamp` (camera time, us), `annotation_timestamp` (us), `boxes_at` ("camera"),
`T_e2g` (4x4 ego -> global at the camera time), `res`, `cam_file` (image path relative to the dataroot), `K` (3x3 intrinsics),
`T_c2e` (4x4 camera -> ego), `image_size` (width, height of the camera image), `radar_poses` (the five radar -> ego 4x4 matrices),
`vectors` (exact polylines in the ego frame per `lane_divider` / `road_divider` / `centreline`),
`boxes` (list, one dict per annotated box in range, below). `training_sample` adds to the "prev" / "next" metas `dt`
(s, signed: negative for prev) and `steps` (list of (dt, ego speed m/s, yaw rate rad/s) per 12 Hz camera frame).

One box of `meta["boxes"]`: `group` ("vehicle" / "pedestrian" / "two_wheeler"), `corners` (4x2 footprint, ego x, y), `center` (3, ego
frame, at the camera time), `size` ([width, length, height] m), `yaw` (rad, ego frame), `vel` (2, world velocity m/s in ego axes),
`vel_source` ("stationary" / "track" / "track+radar" / "radar" / "unknown"), `vel_flag` ("ok", "unobserved", "short_track",
"stationary_but_moving", "radar_override", "radar_conflict"), `vel_sigma` (m/s), `vel_known` (bool), `visibility` (nuScenes
annotator bin "1".."4" = 0-40 / 40-60 / 60-80 / 80-100 %), `token` (annotation), `instance` (instance token), `stationary` (bool:
one fixed world pose, velocity 0), `observed` (bool: has lidar / radar points), `world` (3, global position at the camera time),
`world_ann` (global position at the annotation time), `shift` (2, ego axes, m: what moved the box from annotation to camera time),
`shift_s` (s), `num_lidar_pts` / `num_radar_pts` (int: the nuScenes counts of points in the box in the keyframe sweep, information only),
`share_camera` / `share_radar` / `share_fusion` (0..1: the share of the box footprint the camera sees / the radar fan covers / both;
section 3a).

## 3a. Coverage: what the rig observes, and the fusion share f (v7; decisions D36 - D38)
The rig is one front camera and five radars. The **fusion area** = `camera` cells not `occluded`, inside the `radar` fan; everything
outside it is **don't care on both sides** (training and evaluation). `autogt.coverage` (numpy only, calibration + grid, so a car can
run it live) holds the one definition: `fusion_cells(g)`, `camera_observed(g)`, `coverage_for(g, kind)` (kind `map` / `objects` /
`velocity`: the fusion area; `camera_aux`: camera cells not hidden; `radar_aux`: the radar fan), `camera_cells(...)`, `radar_cells(...)`,
`hidden_cells(...)`, `footprint_share(corners, mask, grid)`, `object_shares(...)`.
- **Hiding is by height**: a box hides the ground behind it with its real volume (its top, not an infinite wall), so low boxes cast short
  shadows (a 0.5 m high box and a camera at 1.6 m: the shadow ends at about 34 m for a box ending at 24 m); `hidden_cells(..., soft=False)`
  is the old 2D rule of v5 / v6 (the study of `bev_fusion/records/footprint_share_study_2026-11-01.md` used it). Only boxes with a lidar or
  radar point hide.
- Boxes stand on the ground plane z = 0 of the ego frame (the plane of the map layers and of `camera`; a box on a hill or a bridge is
  placed on it for this purpose). A point inside a box's footprint is never hidden by that box; a box entirely behind the camera hides nothing.
- **f = `share_fusion`**: the share of the footprint (at least 12 x 12 samples, 0.125 m apart for a car; samples outside the 60 x 52 m
  grid count as not observed) where the camera sees the object (inside the image at the bottom, middle or top of the box height and not
  hidden by another box: a pedestrian above a low car stays visible, also when only the head shows) AND the radar fan covers. The states
  (Mirko, 2026-11-01): care f >= 0.5, partial 0.1 <= f < 0.5, don't care f < 0.1. **One rule for ground truth and detections**: a
  detection gets its f from `shares_in(g, corners, height or group=...)` in the frame of the GT dict `g` (everything it needs is in the
  meta: `K`, `T_c2e`, `image_size`, `radar_poses`, the hiding boxes), the stored `share_fusion` of a GT box is the same function.
  `footprint_share(corners, mask, grid)` is the cheaper ground-plane version without a height (the BEV chat's study of
  `bev_fusion/records/footprint_share_study_2026-11-01.md` used it with the old masks; my copy reproduces its per-object table, v5).
- `rig_radar_poses()`: the five radar poses of the nuScenes rig as matrices (mean over the logs, within 0.06 m and 1.5 degrees), for code
  without a calibration at hand, e.g. the model's precompute step: `radar_cells(gx, gy, rig_radar_poses())`.
- The camera mask is the ground plane z = 0 (what the maps need); the model's `camera_visibility` uses height slices: call
  `camera_cells(gx, gy, K, T_c2e, size, z=<slice height>)` to share the function.

## 4. Rules you must follow when you use or change it
- **Signature = identity of a GT version.** `signature()` is a hash of the six generator files (`grid.py layers.py flow.py
  nuscenes_adapter.py nuscenes_gt.py coverage.py`) plus the options. ANY edit of those files makes every stored GT unreadable by the new code
  (a rebuild of trainval takes 5-8 h) and a running job that imports the live tree keeps its old key. So: never edit them while a job
  runs, tell the owner of the training code before you do, and put the signature next to every number you report. Edits of
  `checks.py view.py video.py report.py cli.py suite.py clean.py selftest.py sets.py` do not change it. `light_nusc.py` (the
  memory-light devkit that `generate` uses) also does not change it, although it decides which records the GT sees: after an edit run
  the selftest, which compares light and full GT. A process keeps the key it computed first (`current_signature()` is memoised): restart
  a reader after an edit. Never compare two models on different signatures.
- **A GT version has a key (decision D29).** Every signature is a version `vN` in `autogt/versions.csv` (`python -m autogt versions`
  lists them, `autogt.current_version()` is the one of this source, the `index.json` of a store carries `gt_version`), and a git tag
  `autogt/vN` sits on the commit that holds that generator. Edit one of the six files and the selftest fails until the table has a row
  for the new signature: add the row, commit, `git tag -a autogt/v<N> -m "<what changed>"`, rebuild the GT, tell the others. To
  reproduce an old GT: `git worktree add temp/gt_v5 autogt/v5`, then run that checkout's `build` (exact command and the check that it is
  the checkout's package that runs: README.md, section Versions; the old tags have no `generate`). Before a
  training starts, `autogt.missing(tokens)` says which keyframes the store lacks (none may be built inside a data-loader worker).
- **Boxes are at the camera time** (decided by the user 2026-10-31, like Waymo's `camera_synced_box`): a moving box of known
  velocity is moved by `v * (t_camera - t_annotation)` (about -36 ms, median 6 cm, max 88 cm); stationary boxes and boxes of
  unknown velocity are not moved. `AutoGT(..., boxes_at="annotation")` / `--boxes-at annotation` restores the annotation positions
  (nuScenes-style evaluation; another signature). A time `dt` is a decoder input only to predict to ANOTHER time (the +-0.5 s keyframes).
- **Stationary means stationary**: a parked car has ONE world pose and velocity 0 in all keyframes; map corrections are allowed per
  map element in world coordinates (`map_offsets`), never per frame.
- **Where a loss or a score counts** (D36 - D38): the coverage of section 3a (`coverage_for`), not the old per-layer habits.
- **Supervision masks**: objects only where `observed`; leave `ignore` out; velocity only where `flow_weight > 0`; `camera` /
  `occluded` / `radar` / `undefined` say what the sensors can see. **"Moving" has one definition**: known velocity and speed above
  1 m/s - cells: `(np.hypot(*g["velocity"]) > 1) & (g["flow_weight"] > 0)`, boxes: `b["vel_known"] and np.hypot(*b["vel"]) > 1`.
  There is no stored mask. `stationary` is something else (the generator's rule: one fixed world pose, v = 0); a slow or creeping
  box is neither stationary nor moving.
  About 80 % of the supervised object cells are static, so a loss over all cells is minimised by predicting 0: weight moving and
  static cells separately and report speed errors split by moving / static with the zero predictor next to them.
- **Map holes**: nuScenes' HD map misses roads built or changed after it was made. In about 0.3 % of the trainval keyframes (45 of
  17,245 scanned; five Boston Pier 4 construction sequences: scene-0499, 0501, 0502, 0515, 0517) there is no road under the ego car and
  the map layers are empty or a single blob; the GT then says "no road" where there is one. Detect it per keyframe with
  `g["drivable_area"][2:8, 50:54].mean() < 0.5` and leave those keyframes out of the map loss. Moving vehicles outside the drivable
  area (1.5 % of them over trainval; 5.9 % in Singapore One-North, 2.0 % Queenstown, 0.8 % Boston; in the two sequences inspected they
  are cars on elevated roads, box bottoms 2-3 m above the ego road) are a limit of the 2D map, not a box error.
- **Honest limits** (full list in README.md): boxes exist at 2 Hz only (yaw is not extrapolated); the 36 ms offset was measured for
  the FRONT camera only; Boston map lines are 0.3-0.7 m off the paint in places; about 22 % of the boxes stand > 0.5 m above or
  below the ego ground plane (hills, overpasses); the occlusion mask is 2D and hides about 11 % of boxes that annotators rate 80-100 %
  visible; radar gives only the radial velocity (the tangential part comes from the annotations); no lidar on disk, so no point-cloud check.
- **A new number needs its GT version**: say VAL or TEST, the signature, and the denominator.

## 5. Recipes
```python
import numpy as np, autogt
g = autogt.load(tok)
moving = (np.hypot(*g["velocity"]) > 1.0) & (g["flow_weight"] > 0)             # moving cells (the one definition, section 4)
cars = [b for b in g["meta"]["boxes"] if b["group"] == "vehicle" and b["observed"] and b["vel_known"] and np.hypot(*b["vel"]) > 1.0]
sup = (g["vehicle"] > 0) & (g["flow_weight"] > 0) & (g["ignore"] == 0)         # supervised vehicle cells
free = (g["drivable_area"] > 0.5) & (g["ignore"] == 0) & (g["undefined"] == 0)
b = g["meta"]["boxes"][int(g["instance"][40, 52])]                              # the box that covers cell (40, 52); instance -1 = none
```
A training sample (previous / current / next keyframe, each the same dict as above):
```python
s = gen.training_sample(tok)             # None at the first / last keyframe of a sequence; needs the generator (not `load`)
cur, nxt = s["cur"], s["next"]           # also s["prev"]; each has its own ego frame (its camera time)
dt = nxt["meta"]["dt"]                   # s from cur to nxt (about +0.5; prev: about -0.5); nxt["meta"]["steps"] = ego motion per camera frame
cur["flow"] * dt                         # first-order displacement of every cell over dt (apparent motion, ego motion included)
# the same object in both keyframes: box["instance"] (instance token); a prediction to dt has the target nxt["vehicle"] etc. in nxt's frame
```
- Look: `python -m autogt view OUT_DIR scene-0061:8 [--total]` writes a 4-panel picture (camera with 3D boxes + map lines, semantic
  BEV, objects + velocity arrows, motion relative to the static world); `python -m autogt sheets OUT_DIR` stacks the ten defaults.
  Pictures and videos are reproducible: delete them with `python -m autogt clean --pictures --videos [--move]`.
- Evaluate a GT version: `python -m autogt report --version v1.0-mini --holdout` -> PDF in `data/autogt/reports/` with every check
  (O1-O8 objects, M1-M3 map, V1-V3 velocity; `checks.py` documents each) and the leave-one-radar-out test of the radar correction.
- Change the generator: edit, `python -m autogt selftest`, `report` on mini, tell the trainer the new signature, then
  `python -m autogt clean --stale-caches data/autogt` (dry run first). A new layer must also be added to the table above (selftest fails otherwise).
- Another dataset: write an adapter that builds `Track`, `Ego`, `RadarFrame`, `GridSpec` for `autogt.flow.estimate_flow`
  (numpy only) and a generator for the static layers (`layers.py` is reusable).

## 6. Files
`suite.py` (`AutoGT`, `load`, `current_signature`), `cli.py` (`generate`, the commands), `sets.py` + `sets/` (scene sets, and the raw-data check),  `versions.py` + `versions.csv` (GT versions), `nuscenes_gt.py` (the generator), `flow.py` (the velocity / flow box, dataset independent),
`nuscenes_adapter.py` (the only code that knows the nuScenes formats), `layers.py` + `grid.py` (exact rasteriser, occlusion, `GridSpec`),
`checks.py` (consistency checks), `report.py`, `view.py`, `video.py` (with `encode`: H.264 with a read-back check), `clean.py`,
`coverage.py` (what the sensors observe, the footprint share), `light_nusc.py` (memory-light devkit), `selftest.py`. Decisions and history: `../bev_fusion/plan/plan.md` section 9.5, `README.md` (validation numbers).
