# GtGenerator (directory `GtGenerator/`) - guide for agents (read this first, README.md is the reference)

The class `GtGenerator` (old name `AutoGT`, still an alias) turns raw nuScenes (map, 3D box annotations, ego poses, radar) into **training-ready BEV ground truth per keyframe**:
static map layers, object footprints, masks, and a velocity / flow field in m/s. Standalone: nothing in it imports `training/`.
Everything is in seconds (never "per frame"). Unit of work = one keyframe (sample token, 2 Hz, 39-41 per sequence; nuScenes mini = 10
sequences / 404 keyframes, trainval = 850 / 34,149).

## 1. Which entry point? (the whole interface)
| I want | do this |
|---|---|
| **the whole GT of one system, one call** (Mirko 2026-11-10) | **BEV Fusion:** `key = GtGenerator.generate_bev_gt("~/data/nuscenes_full", set_name="full")` / `python -m GtGenerator generate-bev ROOT --set full` = the front GT + the lane GT of the rig `bev_fusion` + the packed area masks (intersection, borders 0 and 0.5 m); **Spurmitte:** `GtGenerator.generate_lane_gt(root, rig="lane_center")` / `generate-lane ROOT --rig lane_center` (`rig="all"`: both, a dict). All resumable. They return a **GT key** (`GtGenerator/gt_api.py`): the DATA (`data_identity` of data/NuScenes/manifest.json) + per part the CODE (signature, git commit of GtGenerator/, dirty, version) + the suite release; `str(key)` is one readable line, `key.id` 12 hex characters (same id = same GT), `key.complete`. Not the nuScenes sample token. `GtGenerator.gt_key("bev_fusion", "v1.0-trainval", "full")` reads the key of what is stored, generates nothing |
| to read through a key | `key.load(sample_token)` (BEV Fusion: the front GT), `key.lane(sample_token)` (the lane graph of the system: frame ego for `bev_fusion`, lane for `lane_center`), `key.mask(sample_token, "intersection", border_m=0.5)` = a plain (120, 104) uint8 array in the BEV grid, `[ix, iy]` like the GT maps (x forward, iy = 0 the RIGHT edge), from the packed cache below the lane store (`<lane store>/masks/intersection_b0.5.npz`, 53 MB per border for trainval); the same from a graph without the cache: `GtGenerator.lane_raster.area_mask(g, "intersection", border_m)` (only for the rig `bev_fusion`: the window of the Spurmitte rig does not cover 60 m) |
| **the command line is the class interface** | every `python -m GtGenerator COMMAND` is ONE method of the class `GtGenerator` (GtGenerator/api.py; the command line imports nothing else of the suite; `python -m GtGenerator --help` has the map, `cli.commands()` lists command -> method), the same calls as in Python, `GtGenerator.generate_bev_gt(root)` ...: `generate` / `generate-bev` / `generate-lane` / `area-masks` / `build` make GT, `key SYSTEM` prints the GT key (exit 1 when incomplete, `--json`), `have` / `data` / `sets` / `versions` / `area-mask TOKEN` look at it, `audit-lane SYSTEM` checks a stored lane GT before it is released (exit 1 on a problem; `store_audit.audit_lane`), `view` / `sheets` / `video` / `lanegraph` / `masks` draw it, `check` / `report` / `digest` / `clean` / `selftest [--only cli gt_api generator lane_graph lane_tiers lane_toy clean rig_masks]` check and tidy. `--out` is always the ROOT of the stores (default data/autogt, lane stores below ROOT/lane); wrong arguments exit 2 with a message (AUTOGT_TRACEBACK=1: the traceback), "not there" exits 1; the architecture is drawn in `architecture.svg` (one class, one code block and one output per problem; a test keeps it in step with the code); `versions --update` derives the rows a new signature needs from git; a list of sequences comes from a file (`--scenes-file`) and a key names it `scenes:<id>` |
| the GT of keyframes that are already generated (training, analysis) | `GtGenerator.load(token)` - ~15 ms, no nuScenes needed; the tokens of a sequence: `GtGenerator.scene_tokens("scene-0561", "v1.0-trainval")` (json tables, 1 s) |
| to generate the GT of a nuScenes folder (all sequences or some) and store it where `load` reads it | `GtGenerator.generate("~/data/nuscenes_full")` (or `python -m GtGenerator generate ~/data/nuscenes_full [--out DIR] [--scenes scene-0061 ...] [--stride N]`): finds the version in the folder, memory-light, resumable (a rerun writes only what is missing, a finished group loads no devkit; a rerun that writes nothing leaves the manifest of a finished store alone), returns `{"out", "signature", "gt_version", "version", "scenes", "keyframes", "written", "release", "lane", "lane_rigs", "masks_error"}` (`GtGenerator.generate` adds `keys` and `ok`; its 5th positional argument is `set_name`, `GtGenerator.generate`'s is `stride`: pass keywords); trainval 5-8 h; default `out` = `data/autogt` -> `<signature>/<token>.npz` + `index.json`; one generate / report per folder at a time |
| a small or a full run | `GtGenerator.generate(dataroot, set_name="small")` (the 10 sequences of mini, ~25 min, small experiments) or `"full"` (all 850 of trainval, 5-8 h, releases); shell `--set small|full`; `python -m GtGenerator sets` shows how much of each is stored; the manifest in `index.json` (`gt_version`, `commit`, `dirty`, `sets_complete`, `release`) says what a folder is: `release: true` = complete, released GT version, clean tree, `full` held |
| to know that the raw nuScenes files are all on disk before a long run | `python -m GtGenerator data '~/data/nuscenes_full' [--set small|full]` (exit 1 and the missing files if not; trainval takes minutes); fetching / verifying / linking the raw data is `../data/NuScenes/` (downloads only with the user's OK) |
| to generate single keyframes in code (other options, a quick look) | `GtGenerator(dataroot, version).keyframe(token)` |
| prev / cur / next keyframe + the ego motion between them | `GtGenerator(...).training_sample(token)` (section 5) |
| GT of ALL sequences stored + every check + one PDF | `python -m GtGenerator report --version v1.0-mini` (trainval: `--version v1.0-trainval`, 5-8 h, resumable) |
| to look at a keyframe / drive through a sequence | `python -m GtGenerator view OUT_DIR <token>` (any STORED keyframe, mini or trainval) or `... view OUT_DIR scene-0061:8` (mini, generated now) / `python -m GtGenerator video scene-0061 out.mp4 [seconds]` (mini; a trainval sequence: `python -m GtGenerator video scene-0441 out.mp4 20 --version v1.0-trainval`) |
| to be sure after editing the generator | `python -m GtGenerator selftest` (5 - 10 min; also checks that this file lists every key) |
| to remove what GtGenerator generated | `python -m GtGenerator clean --stale-caches data/autogt` (dry run; `--move` to act, `--delete` only for pictures, videos and reports: a store is deleted by `autoresearch/tools/package.py gt`; a link to D: and a store whose index says `release` are never listed; the caches of released GT versions are left alone, `--include-released` lists them and they can only be moved) |

Python (all of it): `from GtGenerator import generate, GtGenerator, load, scene_tokens, current_signature, GridSpec`.
```python
import GtGenerator
toks = GtGenerator.scene_tokens("scene-0561", "v1.0-trainval")      # 40 sample tokens in time order (json tables only; mini: default version)
g = GtGenerator.load(toks[8])                                        # dict: 24 float32 layers + `instance` (int32) + "meta" (tables below); raises FileNotFoundError
                                                                # (naming the folders that exist) if the token is not stored
gen = GtGenerator.GtGenerator("~/data/nuscenes", "v1.0-mini")             # generate: mini loads in 4 s, the whole devkit is fine
g = gen.keyframe(gen.keyframes("scene-0061")[8])                # same dict, generated now (2-3 s)
# trainval: `GtGenerator(..., light=True, scenes=[...])` still reads the big json tables (minutes on a busy box): prefer `load`
gen.signature() == GtGenerator.current_signature()                   # the cache key (section 4)
```
A cache folder without `index.json`, or with `"complete": false` in it, belongs to a `report` / `build` that is still running or was
interrupted (`build` writes a provisional index with the resolution and supersampling before the first keyframe and the final one at the
end; a resume refuses another resolution): its files are valid, but the sequence list is incomplete. `clean --stale-caches` leaves a
provisional folder alone for an hour after its last write. CLI details: `python -m GtGenerator --help`; the commands you need most are `report`,
`view`, `selftest`.

## 2. Run it (this box is shared)
All commands run **from the repository root (`C:\Auto-research`) inside WSL** with the conda env `gcar`; from Windows prefix `wsl`:
`wsl bash GtGenerator/run_nice.sh -m GtGenerator selftest` (nice 15, one thread; runs any script the same way:
`... run_nice.sh my_script.py`; a running `report` is stopped with `wsl bash -c "pkill -f '[a]utogt report'"`). Rules: one process, no worker pools while a training round runs (`uptime`, `free -m` first),
the full trainval devkit needs 8+ GB (use `light=True, scenes=[...]`; `report` does it automatically), run long jobs in the
background and wait with a bounded loop. Data: `~/data/nuscenes` (mini, 10 sequences, 404 keyframes), `~/data/nuscenes_full`
(trainval, 850 sequences, 34,149 keyframes). Code and output are apart: the code is the package `GtGenerator/` at the repository root (in git); everything
GtGenerator generates is under `data/autogt/` (not in git): stored GT `<signature>/<token>.npz` + `index.json`, reports (PDF)
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
outside it is **don't care on both sides** (training and evaluation). `GtGenerator.coverage` (numpy only, calibration + grid, so a car can
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

## 3a2. Two coverage masks (`rig_masks.py`)
Who uses which: `GtGenerator.rig_masks.mask_for("bev_fusion")` for the project in `bev_fusion/`, `mask_for("lane_center")` for the project in `bev_fusion_lane_center/`; another name is an error.
`GtGenerator.rig_masks.bev_fusion_mask()` (front camera + 5 radars, the BEV-fusion grid, ego frame x forward) and `GtGenerator.rig_masks.lane_mask(res=0.5)` (6 cameras + 5 radars, the Spurmitte window
100 x 100 m, LANE frame x right, y forward): static bool arrays from the calibration, `{"camera", "radar", "fusion"}` (+ `n_cameras` for Spurmitte). No boxes are taken into account:
for BEV fusion the stored front GT has the per-frame version (`camera`, `occluded`, `radar`, `coverage.fusion_cells`); for Spurmitte the six cameras cover 99 % of the window.
`python -m GtGenerator masks out.png` draws both.

## 3b. The lane GT (own version and store; `lane_gt.py`, `lane_graph.py`, `road_graph.py`, `lane_lines.py`)
ONE command makes both: `python -m GtGenerator generate ~/data/nuscenes_full --set full` writes the standard GT (data/autogt/<signature>/) and the lane GT of the same sequences
(data/autogt/lane/<lane signature>/, one store per rig: Spurmitte and BEV Fusion, see l5 below) and stamps the release (`releases.csv`: v10 = front v9 + lane l3, v11 = front v9 + lane l4, v12 = front v9 + lane l5; `--no-lane` for the standard GT only, stride > 1 skips the lane GT).
`python -m GtGenerator generate-lane [--rig lane_center|bev_fusion|all]` makes the lane GT alone. `python -m GtGenerator versions` shows the front versions, the lane versions and the releases.
The lane graph of a keyframe window is NOT part of the front generator: the front GT, its signature and its store do not change when the lane GT does. The graph is made from the
nuScenes map and the ego pose at the front-camera time only (about 40 ms per keyframe); the l4 labels (tiers, visibility) also read the CAM_FRONT ego poses around the keyframe and the
annotated boxes (27 ms). It has its own signature (hash of `lane_graph.py road_graph.py lane_lines.py lane_gt.py lane_tiers.py rig_masks.py` + the parameters that differ from
`lane_gt.DEFAULTS`, the rig is one), its own table `lane_versions.csv` (l1, ...; git tag `autogt/lane-lN`) and its own store `data/autogt/lane/<lane signature>/<sample token>.pkl.gz` + `index.json`.
```
g = GtGenerator.lane_gt.load(sample_token)                                       # stored (python -m GtGenerator generate-lane ~/data/nuscenes_full --set full)
g = GtGenerator.lane_gt.LaneGT("~/data/nuscenes_full", "v1.0-trainval", scenes=[name]).graph(sample_token)       # made on demand
GtGenerator.lane_gt.missing(tokens)                                              # the tokens without a stored graph for this source
```
A graph is a dict: `frame` ("lane": x RIGHT, y forward, ego rear axle at the front-camera time, the lane project's frame; `lane_graph.in_frame(g, "ego")` turns it into x forward,
y left), `half` (50 m: the window is |x|, |y| <= 50), `location`, `sample`, `signature`, `params`, and
- `pieces`: lanes and connectors joined while the join is 1 in, 1 out (a new id only at a split or a merge), cut to the window: `id` (the first 8+ characters of the token
  of its first map piece; "<id>#1" for a second part that comes back into the window: its number depends on the window, so use `piece`, the id of the piece, which is the same
  in every frame, for anything that must be stable), `piece`, `part`, `kind` (lane / connector / mixed), `pts` (n, 2) float32 every 0.5 m in travel order, `start_open` /
  `end_open` (cut by the window), `length`, `tokens` (the nuScenes map pieces it covers), `segments` (`[{token, kind, i0, i1, in_intersection}]`: the index ranges of `pts` per
  map piece, contiguous; `in_intersection` is the test of lane_graph for a connector), `succ` / `pred` (ids inside the window), `left` / `right` / `opposite` (ids of the lanes
  beside it; the opposite lane on either side), `road`, `roads_along`, `lane_count`, `lane_index` (0 = leftmost);
- `events`: `type` split / merge / cross / stop / exit / start / end / window, `pos`, `in` / `out` (piece ids), `dirs_in` / `dirs_out` (unit vectors, one row per id in the same
  order), `partial` (a branch lies outside the window); a stop or exit has `on`, `s` (metres along the part), `n_succ` (stop) / `n_pred` (exit); a window event has `io`
  ("in" / "out"). A stop point is the end of a LANE whose successor connector lies inside an intersection, an exit the start of a LANE whose predecessor connector does (`stop_rule`,
  default: for the intersection branches only). Both can lie inside a joined piece (a lane with one connector after it): then `out` of the stop (`in` of the exit) is that piece;
- `roads`: the roads and intersections of the window (`id`, `kind`, `lane_max`, `pieces`, `succ`, `opposite`); the nuScenes map has no road graph, roads are derived from the
  lanes (road_graph.py docstring);
- `lines`: `lane_divider`, `road_divider`, `boundary` (the outline of the drivable area; `params["boundary_layers"]`), each a list of `{pts, start_open, end_open, corners}` with
  `corners` = `{pos, angle, idx, dir_in, dir_out}` (turning angle over +-2 m above 30 degrees);
- `areas`: `road` (the drivable area, the same layers as the boundary) and `intersection` (the connected areas of the road segments flagged is_intersection, `id` = the node id in
  `roads`, None when no lane connector lies in it), each a list of polygons `{id, exterior, holes, area}` cut to the window. NOT dilated: rasterise a polygon to get the
  intersection pixels as they are, dilate in the head if it wants more.
**l4 (2026-11-07, `lane_tiers.py`, decisions D59 / D60 of the lane project): relevance tiers and camera visibility as LABELS on the same graph** (the geometry, ids, events and polygons are those of l3, checked on every test keyframe).
Per point of every piece (arrays of the length of `pts`): `tier` (uint8) 1 = the ego lane on the path the car really drives (the CAM_FRONT ego poses 25 m before and 55 m after the keyframe: NON-CAUSAL, labels only, never
an input of a model; where the car stands or the sequence ends the path goes on along the map, the straightest successor, to 55 m ahead: `tier_src` 2), 2 = the left / right neighbours of a tier-1 piece (same direction)
within 10 m of a tier-1 point, 3 = the rest (branches not taken, opposite lanes, far lanes); `vis` (bool) = at least one of the six cameras of the rig sees the ground point and no observed vehicle / pedestrian /
two-wheeler box hides it (real box height; the ground UNDER a box is hidden, a rear box hides for the rear cameras; static structures are not in the map, so a point behind a building counts as visible; boxes
are at the annotation time); `occ` (bool) = in a camera's view but hidden by a box; `tier_src` (uint8) 0 / 1 driven / 2 map extension. Per event: `tier`, `vis`. Per graph: `ego_path` = `{"pts" (n, 2) in the frame of the graph,
"back_m", "ahead_m" (the driven metres)}` and `tier_info` (metres per tier, visible metres per tier, hiding boxes, extension metres). New parameters (part of the lane signature): `tier_back` 25, `tier_ahead` 55,
`path_tol` 1.5, `path_cos` 0.7, `neighbour_tol` 10, `min_ahead` 40, `extend` True, `opposite_tier` 3. The cost is about 27 ms per keyframe. Self-tests: `python -m GtGenerator.lane_tiers` (toy lane), `python -m GtGenerator.lane_tiers_selftest` (a fake devkit), both also run by `python -m GtGenerator selftest`. `rig_masks.py` is part of the lane
signature from l4 on (the camera calibration the visibility uses: any edit of it, also of the BEV-fusion mask, is a new lane signature). Known limits of the labels: the cameras see no ground
within about 4.5 m of the car, so `vis` is False there (a consumer that demands nodes near the car must not read it as "hidden"; `occ` is False there); the tier-1 map extension grows whole
pieces (a standing car or the first keyframes of a log get tier 1 for the rest of the piece, more than the 55 m / 25 m it is asked for); `vis` is cameras only, no radar. The graph geometry equals l3
(checked on 970 frames of the partial store by the lane project's scratch check and, for the rig `lane_center` of l5, on 400 stored frames and by the 40 pinned digests of `lane_l4_digest.json`).
**l5 (2026-11-07, Mirko: "BEV Fusion and Spurmitte are different systems, I would like GT for both"): two RIGS of the lane GT, one store each** (`lane_gt.RIGS`, parameter `rig`).
| | `lane_center` (Spurmitte, builds on GaussianCaR; the default, = l4) | `bev_fusion` (BEV Fusion: front camera + radars) |
|---|---|---|
| frame | "lane": x right, y forward | "ego": x forward, y left (the frame of the front GT and of the BEV grid) |
| window | the square \|x\|, \|y\| <= 50 m | the BEV grid: x 0 .. 60 m, y -26 .. 26 m (`g["window"]`, `g["half"]` = 60 is only the bound) |
| `vis` / `occ` | one of the SIX cameras sees the ground point, no box hides it | CAM_FRONT sees it (what the front camera sees the six see too), no box hides it |
| read | `lane_gt.load(token)` | `lane_gt.load(token, rig="bev_fusion")`; made on demand `LaneGT(..., rig="bev_fusion").graph(token)` |
| store | `data/autogt/lane/0702a065c6` (the l4 store: l5 makes the same GT for this rig, checked file by file on stored frames: `same_gt_as` in `lane_versions.csv`, like v9 -> v8) | `data/autogt/lane/<its own signature>` (`python -m GtGenerator versions`) |
Geometry, ids, events, links, lines, areas and the tiers come from the same code; the rig changes the frame, the cut (`lane_graph.clip_rect`: polylines and polygons are cut exactly at the border; a window event lies on the
border; a graph of the rectangular rig carries the keys `window` and `rig`, the `lane_center` graphs have neither: they are the l4 files byte for byte) and the cameras of `vis` / `occ`. The tiers do not depend on the rig (the ego lane
is the ego lane); a BEV Fusion consumer that wants "demanded" nodes reads `vis` of its own camera. `python -m GtGenerator generate` makes the standard GT and both lane stores; `generate-lane --rig bev_fusion` makes one.
`lane_gt.signature()` is the folder of the store, `lane_gt.code_signature()` the hash of the code (they differ where `same_gt_as` points to an earlier store). Release v12 = front v9 + lane l5 (both rigs).
Only vectors and events are stored; rasters are drawn from them where they are needed. Picture: `python -m GtGenerator lanegraph out.png scene-0061:8` (six cameras + top view, rig lane_center).
Open with Mirko, kept as parameters (another value = another lane signature): `boundary_layers`, `corner_deg`, `stop_rule`.

## 4. Rules you must follow when you use or change it
- **Signature = identity of a GT version.** `signature()` is a hash of the six generator files (`grid.py layers.py flow.py
  nuscenes_adapter.py nuscenes_gt.py coverage.py`) plus the options. ANY edit of those files makes every stored GT unreadable by the new code
  (a rebuild of trainval takes 5-8 h) and a running job that imports the live tree keeps its old key. So: never edit them while a job
  runs, tell the owner of the training code before you do, and put the signature next to every number you report. Edits of
  `checks.py view.py video.py report.py cli.py suite.py clean.py selftest.py sets.py` do not change it. `light_nusc.py` (the
  memory-light devkit that `generate` uses) also does not change it, although it decides which records the GT sees: after an edit run
  the selftest, which compares light and full GT. A process keeps the key it computed first (`current_signature()` is memoised): restart
  a reader after an edit. Never compare two models on different signatures.
- **A GT version has a key (decision D29).** Every signature is a version `vN` in `GtGenerator/versions.csv` (`python -m GtGenerator versions`
  lists them, `GtGenerator.current_version()` is the one of this source, the `index.json` of a store carries `gt_version`), and a git tag
  `autogt/vN` sits on the commit that holds that generator. Edit one of the six files and the selftest fails until the table has a row
  for the new signature: add the row, commit, `git tag -a autogt/v<N> -m "<what changed>"`, rebuild the GT, tell the others. To
  reproduce an old GT: `git worktree add temp/gt_v5 autogt/v5`, then run that checkout's `build` (exact command and the check that it is
  the checkout's package that runs: README.md, section Versions; the old tags have no `generate`). Before a
  training starts, `GtGenerator.missing(tokens)` says which keyframes the store lacks (none may be built inside a data-loader worker).
- **A faster generator is not always a new GT**: when the new code makes the same GT byte for byte (`python -m GtGenerator digest` in both trees, then
  `digest --compare`), name the older version in the column `same_gt_as` of `versions.csv`: the store, `load` and `generate` stay on the older
  folder and `gt_version` stays the older key (README.md, "Same GT, new code").
- **Boxes are at the camera time** (decided by the user 2026-10-31, like Waymo's `camera_synced_box`): a moving box of known
  velocity is moved by `v * (t_camera - t_annotation)` (about -36 ms, median 6 cm, max 88 cm); stationary boxes and boxes of
  unknown velocity are not moved. `GtGenerator(..., boxes_at="annotation")` / `--boxes-at annotation` restores the annotation positions
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
import numpy as np, GtGenerator
g = GtGenerator.load(tok)
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
- Look: `python -m GtGenerator view OUT_DIR scene-0061:8 [--total]` writes a 4-panel picture (camera with 3D boxes + map lines, semantic
  BEV, objects + velocity arrows, motion relative to the static world); `python -m GtGenerator sheets OUT_DIR` stacks the ten defaults.
  Pictures and videos are reproducible: delete them with `python -m GtGenerator clean --pictures --videos [--move]`.
- Evaluate a GT version: `python -m GtGenerator report --version v1.0-mini --holdout` -> PDF in `data/autogt/reports/` with every check
  (O1-O8 objects, M1-M3 map, V1-V3 velocity; `checks.py` documents each) and the leave-one-radar-out test of the radar correction.
- Change the generator: edit, `python -m GtGenerator selftest`, `report` on mini, tell the trainer the new signature, then
  `python -m GtGenerator clean --stale-caches data/autogt` (dry run first). A new layer must also be added to the table above (selftest fails otherwise).
- Another dataset: write an adapter that builds `Track`, `Ego`, `RadarFrame`, `GridSpec` for `GtGenerator.flow.estimate_flow`
  (numpy only) and a generator for the static layers (`layers.py` is reusable).

## 6. Files
`api.py` (the static methods of `GtGenerator`: one per command), `suite.py` (`GtGenerator`, `load`, `current_signature`), `cli.py` (the command line: the parser and one `GtGenerator.<method>` call per command), `pipeline.py` (`generate`, `build`), `store.py` (keyframe files, `index.json`, resume checks), `gt_api.py` (`GtKey`, `generate_bev_gt` / `generate_lane_gt`), `lane_raster.py` (packed area masks), `store_audit.py` (the audit of a lane store), `architecture.json` (the drawing), `sets.py` + `sets/` (scene sets, and the raw-data check),  `versions.py` + `versions.csv` (GT versions), `nuscenes_gt.py` (the generator), `flow.py` (the velocity / flow box, dataset independent),
`nuscenes_adapter.py` (the only code that knows the nuScenes formats), `layers.py` + `grid.py` (exact rasteriser, occlusion, `GridSpec`),
`checks.py` (consistency checks), `report.py`, `view.py`, `video.py` (with `encode`: H.264 with a read-back check), `clean.py`,
`coverage.py` (what the sensors observe, the footprint share), `light_nusc.py` (memory-light devkit), `selftest.py`. Decisions and history: `../bev_fusion/plan/plan.md` section 9.5, `README.md` (validation numbers).
