# autogt - automatic ground truth for BEV perception from nuScenes

One standalone tool: it turns the raw nuScenes data (maps, annotations, ego poses, radar) into training-ready GT per keyframe -
static BEV layers, objects, masks, and a **velocity / flow field in m/s** - and checks, shows and caches it. Nothing in it
imports `training/`; `training/gt_bev.py` is only a shim for the older code. Everything is time-based (seconds), never per frame.
**Agents: read `AGENTS.md` first** (entry points, run rules, every layer and box field, the rules for using it); this file is the reference.

**The entry point - read nuScenes from a folder, write the GT of every keyframe** (memory-light, resumable, no GPU):
```python
import autogt                                   # run from the repository root (or put it on PYTHONPATH)
r = autogt.generate("~/data/nuscenes_full")     # all of trainval (5-8 h in one process); out=None -> data/autogt/<signature>/
r = autogt.generate("~/data/nuscenes", set_name="small")  # the small set (the 10 sequences of mini) for small experiments
r = autogt.generate("~/data/nuscenes", out="/data/gt", scenes=["scene-0061"], stride=2)    # a mini sequence, every 2nd keyframe, elsewhere
# -> {"out": folder written, "signature", "gt_version", "version", "scenes", "keyframes": wanted, "written": new files, "release"}; one
#    <sample token>.npz per keyframe + index.json; the version is found in the folder (pass version= if it holds several); a rerun writes
#    only what is missing
g = autogt.load(token)                          # read one back (data="/data/gt" for another folder), no nuScenes needed
```
Same from the shell: `python -m autogt generate ~/data/nuscenes_full [--out DIR] [--version V] [--set small|full | --scenes scene-0061 ...] [--stride N] [--workers N]`.
Parameters of `generate(dataroot, out=None, version=None, scenes=None, stride=1, res=0.5, ss=4, workers=1, verbose=True, set_name=None, **options)`:
`dataroot` the nuScenes folder; `out` where the GT goes (default the place `load` reads); `version` v1.0-mini / v1.0-trainval;
`scenes` names (default all); `set_name` a scene set instead of `scenes` (below); `stride` every n-th keyframe; `res` / `ss` cell size in
m / supersampling; `workers` processes; `options` the generator options of `AutoGT` below (`use_flow`, `use_radar`, `boxes_at`, ...). A
folder written with another resolution or supersampling is never continued; the folder name is the generator signature, so GT of an
older generator is never mixed in.

**Two sets, one manifest.** The scene sets are the files `sets/small.txt` (the 10 sequences of nuScenes mini, 404 keyframes, about 25
minutes: small experiments) and `sets/full.txt` (all 850 sequences of trainval, 34,149 keyframes, 5 - 8 hours: releases); one scene name
per line, `#` starts a comment. `python -m autogt sets` lists them and how much of each the store of the current generator holds
(`autogt.sets.status(name)` in code). Small is a part of full and both share the signature folder, so running small first is not wasted.
The `index.json` of a folder is its manifest: `gt_version` (key of `versions.csv`, only for a released signature), `commit` (short hash of
HEAD when the last run was made) and `dirty` (anything in `autogt/` differed from it), `set` (the set the last run asked for),
`sets_complete` (the sets whose every keyframe has a file; a run takes the sets it can check from its own listing, so a deleted keyframe removes `full` and the release), `environment` (versions of numpy, shapely, opencv, matplotlib, nuscenes-devkit: the generator signature hashes our files, not these) and `release`: true when the folder is complete, its signature is a released
GT version, the tree was clean and `full` is complete. Use a folder with `release: true` for a number you report as a release; a
folder with `dirty: true` or no `gt_version` is scratch. The network input that a model precomputes from the GT (it needs the
backbone) is not made here; it is a product of the model code and carries its own version and the `gt_version` it stands on.

**The whole chain.** get the raw data (`../data/NuScenes/`: its README says what is on D: and which parts are there; its `tools/` list,
download, verify and unpack the archives and link the WSL views `~/data/nuscenes*`; downloads only with the user's OK) -> `python -m autogt
data ~/data/nuscenes_full [--set small|full]` (are all raw files the generator reads on disk: json tables, map expansion, CAM_FRONT, the
five radars; `sets.check_raw`) -> `generate` (above) -> look: `report` (every check, one PDF), `view`, `sheets`, `video`.

One keyframe at a time in code:
```python
from autogt import AutoGT, load
gt = AutoGT("~/data/nuscenes", "v1.0-mini")     # one object (options: use_flow, use_radar, stationary_fix, map_offsets, flow_config, boxes_at)
g = gt.keyframe(sample_token)                   # dict of float32 layers [ix, iy] + meta
s = gt.training_sample(sample_token)            # {"prev", "cur", "next"} + the ego steps between them
gt.signature()                                  # cache key: changes with every edit of the generator or its options
g = load(sample_token)                          # the STORED GT of a keyframe (data/autogt/<current_signature()>/), no nuScenes needed
```
```
python -m autogt selftest                                   # all tests (synthetic world + real keyframes + build)
python -m autogt generate DATAROOT [--out DIR] [--set small|full | --scenes ..] [--stride N] [--workers N]   # THE entry point (above): per-keyframe files DIR/<signature>/
python -m autogt sets                                       # the scene sets small / full and how much of each is stored; `versions` lists the GT versions
python -m autogt build --version v1.0-mini --out DIR [--scenes ..] [--workers N] [--res 0.25]   # its low-level writer (one full devkit, --out required)
python -m autogt check [--holdout]                          # automatic consistency checks (+ leave-one-radar-out radar test)
python -m autogt view OUT_DIR [scene:keyframe ...] [--total]   # sample pictures (4th panel: motion relative to the static world; --total = total flow); `sheets OUT_DIR` stacks the ten defaults
python -m autogt report [--version v1.0-mini] [--holdout] [--samples N]   # ALL sequences in one pass: store the GT (data/autogt/<signature>/), run every check, write ONE PDF
                                                            # (data/autogt/reports/): summary, table per sequence, legend, samples drawn straight into the PDF (no picture files)
python -m autogt clean [--videos] [--pictures] [--reports] [--old-outputs] [--stale-caches DIR | --all-caches DIR] [--include-released] [--older-than DAYS] [--move|--delete]
                                                            # removes only what autogt generated; dry run by default, --move puts it into _to_delete/
python -m autogt video SCENE OUT.mp4 [seconds] [--version v1.0-trainval]   # driving through the GT at 12 Hz (H.264, read back; any sequence, the light devkit of just that one); examples in data/autogt/examples/
```

## Layout
| file | content |
|---|---|
| `flow.py` | **the velocity / flow box** (dataset independent, numpy only): `estimate_flow(tracks, ego, t, grid, radar, config)` - see below |
| `nuscenes_adapter.py` | devkit records -> `Track` / `Ego` / `RadarFrame` (`NuScenesScene`); the only place that knows nuScenes' radar and annotation format |
| `nuscenes_gt.py` | `NuScenesGT`: map layers, objects, masks, velocity / flow layers, training samples, generator signature |
| `layers.py`, `grid.py` | exact supersampled rasteriser (areas, lines, footprints), the old 2D occlusion rule, `GridSpec` |
| `coverage.py` | **what the front rig observes** (numpy only, calibration + grid): `camera_cells`, `radar_cells`, `hidden_cells` (boxes hide by their height), `object_shares` / `footprint_share` (the share f of an object in the fusion area), `fusion_cells`, `coverage_for` - see "Coverage" below |
| `lane_gt.py`, `lane_graph.py`, `road_graph.py`, `lane_lines.py` | **the lane GT** (own version, not part of the front generator; numpy + shapely, the nuScenes map only): the lane graph with ids, successors and events, neighbours / lane counts / roads, dividers + road boundary with corners (section Lane GT); `lane_versions.csv` its versions; `lane_graph_view.py` draws it (`python -m autogt lanegraph`), `lane_graph_selftest.py` tests it |
| `rig_masks.py` | **the two coverage masks**, one per sensor rig, static from the calibration: `bev_fusion_mask()` (front camera + 5 radars, the BEV-fusion grid) and `lane_mask()` (6 cameras + 5 radars, the Spurmitte window); `python -m autogt masks out.png` draws both |
| `versions.py`, `versions.csv` | the GT versions (D29): the table (version key, signature, tag, what changed), `version_of`, the key stamped into `index.json`; `code_state()` (commit + dirty flag of the manifest) |
| `sets.py`, `sets/small.txt`, `sets/full.txt` | the scene sets (one scene name per line): `names`, `scenes`, `tokens`, `status`, `complete_in`; `generate(set_name=...)`, `python -m autogt sets`; and `check_raw(dataroot, ...)`: are the raw files the generator reads on disk, `python -m autogt data` |
| `suite.py`, `cli.py`, `__main__.py` | `AutoGT` facade, `load`, `current_signature`, **`generate`** (cli.py: the entry point, groups of 60 sequences with the light devkit, `detect_version`); commands `selftest`, `build` (parallel, resumable cache), `check`, `view`, `sheets`, `video`, `report`, `clean` |
| `run_nice.sh` | runs any module or script at nice 15 with one thread in the gcar env with `import autogt` working: `wsl bash autogt/run_nice.sh -m autogt selftest` |
| `AGENTS.md` | the short guide for agents; `selftest` fails if it does not name every layer, meta entry and box field |
| `checks.py`, `view.py`, `video.py`, `report.py`, `clean.py`, `selftest.py` | the tools behind the commands (`video.py` also holds `encode`: H.264 video with a read-back check; mp4v does not play in browsers or the Claude app) |

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

## Two coverage masks: BEV fusion and Spurmitte (`rig_masks.py`)
The two projects have different rigs and so different coverage. BEV fusion sees with the front camera and the five radars on the front grid (x forward 0 - 60 m, y +-26 m): the static
mask covers 63 % of the grid with the camera, 100 % with the radar fans, the fusion area (camera and radar) 63 %; with the boxes that hide cells it is the per-frame mask of the stored
GT (`camera`, `occluded`, `radar`, layers, `coverage.fusion_cells`). Spurmitte (lane centres on GaussianCaR features) works on a 100 x 100 m window round the ego with six cameras and
five radars: the six cameras together cover 99 % of that window on the ground plane, the radars 100 %, so no region of the window is unseen by geometry alone; what is unseen there is
what boxes hide, and the static Spurmitte mask does not know that. Both masks come from the calibration alone (the mean of the 850 logs; the logs differ by at most 0.04 m and 1 - 3 degrees),
nothing is stored, there is no GT version; the same functions as in `coverage.py`. The static front camera mask equals the per-frame layer of the stored GT with IoU 0.993, the radar 0.9999.
Lane frame arrays: [ix, iy] with x RIGHT and y forward, cell centres at -50 + (i + 0.5) res.

## Coverage (`coverage.py`, v7; decisions D36 - D38 of the BEV project, `bev_fusion/records/sensor_coverage_2026-11-01.md`)
The rig is one front camera and five radars, so the GT says per cell and per object what the rig can observe. **Fusion area** = `camera`
cells that are not `occluded`, inside the `radar` fan (D37: fusion numbers are reported there); outside it the cell is **don't care** on
both sides, in training and in evaluation (D38). `coverage_for(g, kind)` gives the area of an output type: map layers, objects and
velocity on the fusion area, the camera aux head on the camera cells not hidden, the radar aux head on the fan.
- **Boxes hide with their height** (v7): a box stands on the ground plane z = 0 (the plane of the map layers) and hides the ground behind
  it with its real volume: the line from the camera to the cell must pass through the box's footprint at a height between 0 and its top
  (`blocked`, `hidden_cells`). A 0.5 m high box and a camera at 1.6 m cast a shadow of about 10 m behind a box that ends at 24 m; v5 / v6 took every box as an infinitely high wall (16 % of the
  objects of the stored val frames lost at least half of their share to it). Only boxes with a lidar or radar point hide. `camera` (the ground
  plane z = 0 projected into the image), `radar` (the fan, 99.9 % of all detections inside) and `undefined` are unchanged.
- **The footprint share f** (D38 b), per box in the meta as `share_fusion` (and `share_camera`, `share_radar`): the share of the footprint
  (at least 12 x 12 samples, 0.125 m apart for a car; samples outside the 60 x 52 m grid count as not observed) where the camera sees the
  object and the radar fan covers. The camera sees a sample when it sees the object above it at the bottom, middle or top of its height
  (inside the image, not hidden by another box), so a pedestrian above a low car stays visible (also when only the head shows). Mirko's
  first states: care f >= 0.5, partial 0.1 <= f < 0.5, don't care f < 0.1. One rule for ground truth and detections:
  `shares_in(g, corners, height)` gives a detection its f in the frame of a GT dict (the meta holds `K`, `T_c2e`, `image_size`,
  `radar_poses` and the hiding boxes); the stored `share_fusion` of a GT box is the same function. `footprint_share(corners, mask, grid)`
  is the cheaper ground-plane version without a height: with the OLD masks (`hidden_cells(soft=False)`) it reproduces the per-object
  table of the BEV chat's study (`bev_fusion/records/footprint_share_study_2026-11-01.md`, 48,724 objects of the val frames).
- `num_lidar_pts` / `num_radar_pts` per box are the nuScenes counts, stored as information (Mirko: no per-box radar gate).
- To share "the camera sees the cell" with a model: `camera_cells(gx, gy, K, T_c2e, size, z=<height of the slice>)`.

## Lane GT (`lane_gt.py`, `lane_graph.py`, `road_graph.py`, `lane_lines.py`; own version l1 ..., asked for by the lane project, Mirko 2026-11-04 "GT chat builds it")
One window of 100 x 100 m around the ego (rear axle at the front-camera time) in the lane project's frame (x right, y forward; `in_frame(g, "ego")` for x forward, y left), built
from the nuScenes map only (no image, radar or box). It is NOT part of the front generator: the front GT, its signature and its store are unchanged by it, and the lane GT has its
own signature, table (`lane_versions.csv`, git tag `autogt/lane-lN`) and store (`data/autogt/lane/<lane signature>/<token>.pkl.gz`, `python -m autogt generate-lane`). What a
graph holds is listed in `AGENTS.md`, section 3b.
- PIECES. A lane or connector is a map piece; chains of map pieces are joined while every join is 1 in, 1 out, so an id changes only at a split or a merge. EVENTS at the joins:
  split, merge, cross (n in, n out), start / end (no predecessor / successor in the map), window (the polyline crosses the border; per keyframe) and STOP: the end of a lane
  whose successor connector lies inside an intersection (`road_segment.is_intersection`; 80 - 84 % of the connectors do, 0.2 - 1.5 % of the lanes), and its mirror EXIT: the start of
  a lane whose predecessor connector does. With several successors the stop is for the intersection branches only (`stop_rule`, open with Mirko). Pass-through joins are not events.
  Per piece `segments` gives the index range of its points per map piece with the intersection flag, so a consumer can drop the points of connectors inside intersections and keep
  the connectors outside (16 - 20 % of the connectors are plain continuations). A part cut by the window at both ends is dropped when shorter than `min_len`; a short part with a
  natural end stays, so links and events never dangle.
- ROADS. nuScenes has no road graph (a lane's edge-line tokens match no road block in 0 of 3,662 lanes; in singapore-queenstown and -hollandvillage the road segments are only
  intersections and the road blocks do not point at a segment), so neighbours, lane counts and roads are derived from the lanes (`road_graph.py` docstring). Check: of the
  730 lane pairs that share a divider node on nuScenes mini, 723 are found as right neighbours and none as left.
- LINES. `lane_divider`, `road_divider` and the BOUNDARY (the outline of the union of the drivable-area polygons; nuScenes has no boundary record) cut to the window, with corners.
- AREAS. The road area (inside the boundary) and the intersection areas (the road segments flagged is_intersection, one id per connected area, matching the nodes of `roads`) as
  polygons, not dilated: the head draws road and intersection pixels from them.
- CHECKS (`lane_graph_selftest.py`, in `selftest`): a made-up map (split, merge, stop, join, window cut, frames), the points lie on autogt's own centre-lines (median 0.01 m, 99 % 0.05 m
  both ways), the junction events rebuild the map's junctions (F1 0.99), the neighbours against the map's divider tokens, corners and chunked outlines, the store (generate, resume,
  load, another parameter = another folder). Against the lane project's own `make_graph_gt.py` (same frame and window) 47 of 107,538 points differ by more than 0.3 m (pieces
  shorter than 2 m, which this code drops).
- COST. About 40 ms per keyframe; the map graphs are made once per location and process.

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
- the box shift v * dt (a one-off script, commit 769aafc: `bev_fusion/temp/checks/radar_box_shift.py`; the selftest keeps the sign check against the interpolated annotation; mini, 340 moving observed vehicles >= 5 m/s, 5..45 m ahead; raw radar
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
- the occlusion mask is 2D: it also hides about 11 % of boxes that annotators rate as 80-100 % visible (pedestrians behind cars)
- radar gives only the radial velocity; the tangential part comes from the annotations. Not validated against lidar scene flow
  (no lidar on disk)
- `coverage.shares_in` (one rule for the share f of a ground-truth box and of a detection) counts every observed box of the frame as a possible occluder, also the one that the
  detection IS. The GT shares are not affected (a box does not hide itself). A caller that scores a detection against its matched box passes a frame dict whose
  `meta["boxes"]` leaves that box out; a fix inside coverage.py would change the front signature, i.e. the key of the BEV GT cache, so it is not made (decided 2026-11-04)
- the light devkit scanned the three big tables (sample_data 17 s, ego_pose 7 s, sample_annotation 6 s) for every group of 60 sequences, 35 s of the 2606 s of the lane GT run per group.
  `light_nusc.ensure_cache` now reads them once (92 s, 277 MB in data/autogt/cache/light_<version>/, one file per sequence, valid for tables of the same size and time) and a group
  comes from the cache in 3.7 s; `generate` and `generate-lane` make it for trainval before they start. Checked: the cached tables equal the scanned ones record for record

## Versions (decision D29, 2026-11-01)
The GT may change, but every state of the generator is a **version** with a key, so that an old GT and the training on it can be
reconstructed. The signature (a hash of the six generator files + options) names the folder of the stored GT; the version key `vN`
is its human name. A version is three things: a row in `versions.csv` (key, signature, tag, date, generator, what changed), a git
tag `autogt/vN` on the commit that holds that generator, and `gt_version` in the `index.json` of the stored GT.
`python -m autogt versions` lists them; `autogt.current_version()` is the key of this source; `autogt.missing(tokens)` tells before a
training which keyframes the store lacks.

| key | signature | what it is |
|---|---|---|
| v1, v2 | 85f096748f, 962e788566 | the generator before the suite, `training/gt_bev.py` of the commits 7f5cb10 / 7c8f12e; its GT is in the trainer's cache `bev_fusion/data/cache/gt` |
| v3 | 7fe1149566 | the standalone suite, moving boxes at the annotation time |
| v4 | fefd519974 | boxes at the camera time, first state (superseded after 18 minutes) |
| v5 | 15efbde2d1 | review fixes; the generator of the t10 runs (GT stored for all of trainval) |
| v6 | 02a1217742 | generator fixes: ego velocity at the first / last pose of a sequence, instance / velocity / flow_sigma on the same sub-pixels, `ego_half_window` |
| v7 | b84e8ac9b2 | coverage (D36 - D38): boxes hide by their height (softer `occluded`), `coverage.py`, per box `share_camera` / `share_radar` / `share_fusion` and the nuScenes point counts; `camera`, `radar`, `undefined` unchanged. Never built for trainval (superseded by v8 before it finished) |
| v8 | c643a7539d | v7 made fast, the same GT (bit-identical on 34 keyframes of four maps): radar files read in parallel, map polygons and centre-line points made once, the box shares in one pass, `keyframe_gt` split into methods (the first trainval store of the coverage GT) |
| v9 | 6a7aa7864a | v8 made twice as fast, the same GT (211 frames of all ten mini sequences byte for byte equal to v7 / v8): the map's record-in-patch test with the geometry made once, `inside_quad` for four edges at once; it shares the store of v8 (`same_gt_as`); **current** |

**Same GT, new code.** A version whose code makes the same GT as an older one byte for byte (faster or tidier code) names it in the column
`same_gt_as` of `versions.csv` (v9 names v8) and needs no store of its own: `load`, `missing`, `generate` and `have` use the folder of the older
version, `gt_version` in its manifest is the older key (the key names the GT, the signature the code; the manifest's `generator` names the code that
wrote it). The proof is `python -m autogt digest A.json` in the source tree of each version (a worktree of the older tag and the new tree) and
`python -m autogt digest --compare A.json B.json`: a digest per frame, every array byte and meta entry, of every other keyframe of all ten mini sequences and
a few 12 Hz frames.

**When you edit one of the six files** (`grid.py layers.py flow.py nuscenes_adapter.py nuscenes_gt.py coverage.py`) the signature changes and
`python -m autogt selftest` fails until `versions.csv` has a row for it: add the row, commit, `git tag -a autogt/v<N> -m "..."`, rebuild
the GT, tell the others. **To rebuild an old version** (v3, v4, v5): `git worktree add temp/gt_v5 autogt/v5`, then
`cd temp/gt_v5 && PYTHONPATH=bev_fusion python -m autogt build --dataroot <nuScenes folder> --version v1.0-trainval --out <folder>`
(`build`, not `generate`: the tags hold the package under `bev_fusion/autogt/` and have the full-devkit writer only, 8+ GB for
trainval; the tags v3 - v5 predate `generate`). A `PYTHONPATH` that does not name the checkout silently imports the live package, so check
first: `PYTHONPATH=bev_fusion python -c "import autogt; print(autogt.__file__)"` must print a path inside `temp/gt_v5`. Then `<folder>` holds
`<signature>/` of that version (v5: 15efbde2d1). v1 and v2 are `bev_fusion/training/gt_bev.py` at their tag (`NuScenesGT.keyframe_gt`,
no `generate`). Tags are pushed with the code (`git push origin autogt/v1 ...`).

## Releases: the suite as one number (`releases.csv`)
`python -m autogt generate` makes the standard nuScenes GT (the front GT, signature and versions as above) and the lane GT (own signature, `lane_versions.csv`) of the same
sequences, so the pair carries ONE number: the release. The numbers are one running sequence with the front versions: `v10` = front v9 + lane l3 (2026-11-04), every later
change of the front GT or of the lane GT is the next number (v11, ...; the next front row of versions.csv is v11, v10 is taken by the release). The front store is not touched by the lane GT (the BEV training and evaluation key their GT cache on the front
signature), and the manifest of a front store records `suite_release`. `python -m autogt selftest` fails when the pair (front version, lane version) of this source has no row in
`releases.csv`. `python autoresearch/tools/package.py gt` knows both families of stores: the latest front version and the latest lane version stay.

## Adding a dataset
Write an adapter that builds the four inputs of `estimate_flow` (`Track`, `Ego`, `RadarFrame`, `GridSpec`) and a generator for
the static layers; `layers.py` and `flow.py` are reusable as they are.

## History
The generator was `training/gt_bev.py` (+ the first flow version `training/gt_flow.py`) and about twenty one-off scripts in
`temp/checks/` (projection checks, pitch / time-offset tests, Boston debug, map-alignment per location, viewers, videos); they
are folded into this suite and removed - git history before the commit "autogt suite" has them. What they found is in
`../bev_fusion/plan/plan.md` section 9.5.

## Using the velocity targets in a loss (note from the t1_r4_soa review, 2026-10-30)
About 80 % of the supervised object cells are static (velocity 0) and 20 % move: an L1 / MSE over all object cells is minimised by
predicting zero, and the gradient of a dense motion head at the BEV feature was measured ~18x below the objects group. Balance the
two groups (weight moving cells, `|velocity| > 1 m/s`, separately from static ones, or use a loss normalised per group), use
`flow_weight` (0 = unknown) as the mask and `flow_sigma` as an uncertainty (e.g. a Huber / beta-NLL scale), and report speed errors
split into moving and static cells with the zero predictor next to them. The signal is in the input: the median radar Doppler inside
a moving box reproduces the GT radial speed (corr 0.989, review notes in `bev_fusion/temp/review/observe_2026-10-30/NOTES.md`). A `moving`
mask is not stored in the GT on purpose (it would change the generator signature and invalidate the caches); derive it from
`velocity`.
