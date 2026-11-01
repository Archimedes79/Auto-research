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
`sets_complete` (the sets whose every keyframe has a file) and `release`: true when the folder is complete, its signature is a released
GT version, the tree was clean and `full` is complete. Use a folder with `release: true` for a number you report as a release; a
folder with `dirty: true` or no `gt_version` is scratch. The network input that a model precomputes from the GT (it needs the
backbone) is not made here; it is a product of the model code and carries its own version and the `gt_version` it stands on.

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
python -m autogt video SCENE OUT.mp4 [seconds]              # driving through the GT at 12 Hz (H.264, read back); the examples so far: scene-0061, scene-1077, scene-0655 in data/autogt/examples/
```

## Layout
| file | content |
|---|---|
| `flow.py` | **the velocity / flow box** (dataset independent, numpy only): `estimate_flow(tracks, ego, t, grid, radar, config)` - see below |
| `nuscenes_adapter.py` | devkit records -> `Track` / `Ego` / `RadarFrame` (`NuScenesScene`); the only place that knows nuScenes' radar and annotation format |
| `nuscenes_gt.py` | `NuScenesGT`: map layers, objects, masks, velocity / flow layers, training samples, generator signature |
| `layers.py`, `grid.py` | exact supersampled rasteriser (areas, lines, footprints), occlusion mask, `GridSpec` |
| `versions.py`, `versions.csv` | the GT versions (D29): the table (version key, signature, tag, what changed), `version_of`, the key stamped into `index.json`; `code_state()` (commit + dirty flag of the manifest) |
| `sets.py`, `sets/small.txt`, `sets/full.txt` | the scene sets (one scene name per line): `names`, `scenes`, `tokens`, `status`, `complete_in`; `generate(set_name=...)`, `python -m autogt sets` |
| `suite.py`, `cli.py`, `__main__.py` | `AutoGT` facade, `load`, `current_signature`, **`generate`** (cli.py: the entry point, groups of 60 sequences with the light devkit, `detect_version`); commands `selftest`, `build` (parallel, resumable cache), `check`, `view`, `sheets`, `video`, `report`, `clean` |
| `run_nice.sh` | runs any module or script at nice 15 with one thread in the gcar env with `import autogt` working: `wsl bash autogt/run_nice.sh -m autogt selftest` |
| `AGENTS.md` | the short guide for agents; `selftest` fails if it does not name every layer, meta entry and box field |
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
- the occlusion mask is 2D: it also hides about 12 % of boxes that annotators rate as 80-100 % visible (pedestrians behind cars)
- radar gives only the radial velocity; the tangential part comes from the annotations. Not validated against lidar scene flow
  (no lidar on disk)

## Versions (decision D29, 2026-11-01)
The GT may change, but every state of the generator is a **version** with a key, so that an old GT and the training on it can be
reconstructed. The signature (a hash of the five generator files + options) names the folder of the stored GT; the version key `vN`
is its human name. A version is three things: a row in `versions.csv` (key, signature, tag, date, generator, what changed), a git
tag `autogt/vN` on the commit that holds that generator, and `gt_version` in the `index.json` of the stored GT.
`python -m autogt versions` lists them; `autogt.current_version()` is the key of this source; `autogt.missing(tokens)` tells before a
training which keyframes the store lacks.

| key | signature | what it is |
|---|---|---|
| v1, v2 | 85f096748f, 962e788566 | the generator before the suite, `training/gt_bev.py` of the commits 7f5cb10 / 7c8f12e; its GT is in the trainer's cache `bev_fusion/data/cache/gt` |
| v3 | 7fe1149566 | the standalone suite, moving boxes at the annotation time |
| v4 | fefd519974 | boxes at the camera time, first state (superseded after 18 minutes) |
| v5 | 15efbde2d1 | review fixes; the generator of the t10 runs; **current** |

**When you edit one of the five files** (`grid.py layers.py flow.py nuscenes_adapter.py nuscenes_gt.py`) the signature changes and
`python -m autogt selftest` fails until `versions.csv` has a row for it: add the row, commit, `git tag -a autogt/v<N> -m "..."`, rebuild
the GT, tell the others. **To rebuild an old version** (v3, v4, v5): `git worktree add temp/gt_v5 autogt/v5`, then
`cd temp/gt_v5 && PYTHONPATH=bev_fusion python -m autogt build --dataroot <nuScenes folder> --version v1.0-trainval --out <folder>`
(`build`, not `generate`: the tags hold the package under `bev_fusion/autogt/` and have the full-devkit writer only, 8+ GB for
trainval; the tags v3 - v5 predate `generate`). A `PYTHONPATH` that does not name the checkout silently imports the live package, so check
first: `PYTHONPATH=bev_fusion python -c "import autogt; print(autogt.__file__)"` must print a path inside `temp/gt_v5`. Then `<folder>` holds
`<signature>/` of that version (v5: 15efbde2d1). v1 and v2 are `bev_fusion/training/gt_bev.py` at their tag (`NuScenesGT.keyframe_gt`,
no `generate`). Tags are pushed with the code (`git push origin autogt/v1 ...`).

## Adding a dataset
Write an adapter that builds the four inputs of `estimate_flow` (`Track`, `Ego`, `RadarFrame`, `GridSpec`) and a generator for
the static layers; `layers.py` and `flow.py` are reusable as they are.

## History
The generator was `training/gt_bev.py` (+ the first flow version `training/gt_flow.py`) and about twenty one-off scripts in
`temp/checks/` (projection checks, pitch / time-offset tests, Boston debug, map-alignment per location, viewers, videos); they
are folded into this suite and removed - git history before the commit "autogt suite" has them. What they found is in
`../bev_fusion/plan.md` section 9.5.

## Using the velocity targets in a loss (note from the t1_r4_soa review, 2026-10-30)
About 80 % of the supervised object cells are static (velocity 0) and 20 % move: an L1 / MSE over all object cells is minimised by
predicting zero, and the gradient of a dense motion head at the BEV feature was measured ~18x below the objects group. Balance the
two groups (weight moving cells, `|velocity| > 1 m/s`, separately from static ones, or use a loss normalised per group), use
`flow_weight` (0 = unknown) as the mask and `flow_sigma` as an uncertainty (e.g. a Huber / beta-NLL scale), and report speed errors
split into moving and static cells with the zero predictor next to them. The signal is in the input: the median radar Doppler inside
a moving box reproduces the GT radial speed (corr 0.989, review notes in `bev_fusion/temp/review/observe_2026-10-30/NOTES.md`). A `moving`
mask is not stored in the GT on purpose (it would change the generator signature and invalidate the caches); derive it from
`velocity`.
