# GtGenerator - guide for agents

Training-ready ground truth per nuScenes keyframe (sample token, 2 Hz, 39 - 41 per sequence; mini = 10 sequences / 404 keyframes, trainval = 850 / 34,149): static map layers, object footprints, masks and
a velocity / flow field in m/s (**front GT**), and the **lane GT**. Everything is in seconds, never "per frame". Nothing in it imports a problem's code. Commands, stores, versions and how to extend
it: [`README.md`](README.md). Current: front GT v13 (signature `a107e0b4e8`), lane GT l6.1 (stores `9ad1d4ce30` rig `lane_center`, `8a04143278` rig `bev_fusion`), release v15.

## 1. The interface
A problem (`problems/`), the research core (`autoresearch/`) and `datasets/` use `import GtGenerator` and the names of this table (they are `GtGenerator.__all__`; `__init__.py` has the contract line of
each), or `python -m GtGenerator COMMAND` across a process boundary, and nothing below them: no module or file of this folder, no folder or file name below the store root, no file format, no version table. A name that is missing is added
to the table, not imported from below. The test of the rule: `GtGenerator.check_imports([folder, ...])` lists every use below the interface as `file:line: text` (`[]` = none; one line in a test of a
problem), and `python -m GtGenerator selftest --only interface` runs it over the three folders.

| I want | name | what comes back |
|---|---|---|
| where the stores are | `GtGenerator.store_root()` | str: `<data root>/groundtruth`, resolved at every call |
| which GT this source reads (for a cache name, a record) | `GtGenerator.current_signature()` (front), `GtGenerator.lane_signature(rig="lane_center")`; on a generator object `gen.store_signature()` | str, 10 hex: the key of the STORE (after `same_gt_as`), never the code signature: an edit that changes no GT keeps it |
| the version name of it | `GtGenerator.current_version()`, `GtGenerator.version_of_key("0.5_4_a107e0b4e8")`, `GtGenerator.version_names()` | "v13"; None for a key that is no released version; the list of all front versions, oldest first (what a split may name as its GT) |
| the check before a training | `GtGenerator.gt_key("bev_fusion", "v1.0-trainval", "full")`; shell `python -m GtGenerator key bev_fusion --set full` (exit 1 when incomplete); `GtGenerator.missing(tokens)` / shell `have` for the front GT alone | `key`, a `GtGenerator.GtKey`: `key.complete`, `key.command` (the shell line that makes the parts that are missing, "" when none is), `key.id` (the same GT has the same id), `print(key)`; the list of tokens that are not stored |
| to read through a key | `key.load(token)`, `key.lane(token)`, `key.mask(token, "intersection", border_m=0.5)` | the front keyframe; the lane graph (frame ego for `bev_fusion`, lane for `lane_center`); (120, 104) uint8, `[ix, iy]` like the GT maps (`bev_fusion` only) |
| a stored front keyframe | `GtGenerator.load(token)` | the dict of section 3 (no nuScenes needed; FileNotFoundError names the folders that exist) |
| a stored lane graph | `GtGenerator.load_lane(token, rig="lane_center")`, `GtGenerator.lane_tokens(rig="lane_center")` | the dict of section 3b; the set of the sample tokens that are stored |
| the keyframes of a sequence | `GtGenerator.scene_tokens("scene-0561", "v1.0-trainval")` | list of sample tokens in time order (json tables only, no devkit) |
| GT made in a loader, single keyframes | `GtGenerator.GtGenerator(root, "v1.0-mini")` = `gen` (everything after the version by keyword: `nusc=` an open devkit, `light=True, scenes=[...]` the memory-light one; options `use_flow`, `use_radar`, `flow_config`, `boxes_at`) | `gen.keyframe(token)`: the dict of section 3; `gen.training_sample(token)`: prev / cur / next (section 4); `gen.keyframes(scene)`, `gen.scenes()`; the raw side the GT itself uses: `gen.cam(token)`, `gen.camera_track(token)`, `gen.ego_steps(t0, t1, track)`, `gen.nusc` |
| the whole GT of a system, or of a nuScenes folder | `GtGenerator.generate_bev_gt(root, set_name="full")` (front GT + lane GT of the rig `bev_fusion` + intersection masks at borders 0 and 0.5 m), `GtGenerator.generate_lane_gt(root, rig="lane_center")` (`rig="all"`: both, a dict), `GtGenerator.generate(root, set_name=..., scenes=..., workers=...)` (everything); shell `generate-bev`, `generate-lane`, `generate` | a GT key (`generate`: a dict); resumable, memory-light, no GPU, trainval takes hours. Pass keywords: the method `GtGenerator.GtGenerator.generate` (adds `keys`, `ok`) orders its arguments differently |
| raw nuScenes as the GT reads it | `GtGenerator.light_nuscenes(root, version, scenes=[...], cameras=(...))`, `GtGenerator.pose_matrix(record)` | a devkit-like object holding only these sequences and sensors; (4, 4) float64 of an ego_pose / calibrated_sensor record |
| the definitions a network and its tests share with the GT | `GtGenerator.GridSpec`, `GtGenerator.AREAS`, `GtGenerator.LINES`, `GtGenerator.OBJECTS`, `GtGenerator.DIST_MAX`, `GtGenerator.cell_centres(grid)`, `GtGenerator.occlusion(gx, gy, cam_xy, footprints)`, `GtGenerator.radar_cells(gx, gy, poses)` | the grid (section 3), the layer names, the truncation of the distance layers in m, the cell centres `(gx, gy)`, bool cell masks (hidden behind a footprint; inside a radar fan) |
| where a loss or a score counts | `GtGenerator.coverage_for(g, kind)`, `GtGenerator.shares_in(g, corners, height or group=...)`, `GtGenerator.mask_for("bev_fusion")` | section 3a: bool cells; the observed shares of a box; the static masks of a rig |
| a camera picture with the lane graph on it | `GtGenerator.lane_camera_image(gt, sample, graph, channel, T_e2g_front)` | (H, W, 3) uint8 BGR |
| to look | `python -m GtGenerator view OUT_DIR <token>` (stored) or `scene-0061:8` (mini, made now); `video SCENE OUT.mp4`; `lanegraph OUT.png scene-0061:8`; `report` (all checks, one PDF) | files |
| to be sure after editing the generator | `python -m GtGenerator selftest` (it also checks that this file names every layer, meta entry and box field, and every public name) | "ok" lines, or the assertion that failed |

```python
import GtGenerator
toks = GtGenerator.scene_tokens("scene-0561", "v1.0-trainval")    # 40 sample tokens in time order (json tables only)
g = GtGenerator.load(toks[8])                                      # the stored keyframe: layers + "meta"
lanes = GtGenerator.load_lane(toks[8])                             # the stored lane graph (Spurmitte; rig="bev_fusion": BEV Fusion's frame and window)
gen = GtGenerator.GtGenerator("~/data/nuscenes", "v1.0-mini")      # keyword options: use_flow, use_radar, flow_config, boxes_at
g = gen.keyframe(gen.keyframes("scene-0061")[8])                   # the same dict, made now (seconds)
gen.signature(), gen.store_signature()                             # the code / the store it fills and reads = GtGenerator.current_signature() (they differ only under `same_gt_as`)
```

## 2. Running it (the box is shared)
From the repository root in WSL, conda env `gcar`; `wsl bash GtGenerator/run_nice.sh -m GtGenerator ...` (nice 15, one thread; runs any script the same way). One process and no worker pools while a
training runs (`uptime`, `free -m` first). The full trainval devkit needs 8+ GB: use `GtGenerator.GtGenerator(..., light=True, scenes=[...])` (it still reads the big json tables) or `load`. Run long jobs
in the background; one writer per store folder. Raw nuScenes (`~/data/nuscenes` mini, `~/data/nuscenes_full` trainval) is never written to; stores are derived data below `<data root>/groundtruth`.
A front store whose `index.json` is missing or says `"complete": false` belongs to a run that is still going or was interrupted: its files are valid, its sequence list is not.

## 3. A front keyframe
Grid: ego frame at the **front-camera time**, x forward 0 .. 60 m, y left -26 .. 26 m, 0.5 m cells (`GridSpec(res=0.25)` for 0.25 m): arrays `[ix, iy]` of shape (120, 104); `iy = 0` is the RIGHT edge,
cell centre x = (ix + 0.5) * res, y = -26 + (iy + 0.5) * res. Vectors are (2, nx, ny) = (x, y) in ego axes. 24 float32 layers (stored as float16) + `instance` (int32) + `meta`.

| key | shape | meaning |
|---|---|---|
| `drivable_area` `ped_crossing` `walkway` `stop_line` `carpark_area` | (nx, ny) | map areas, coverage fraction 0 .. 1 (exact 4 x 4 supersampling) |
| `lane_divider` `road_divider` | (nx, ny) | 1-cell line masks |
| `lane_divider_dist` `road_divider_dist` | (nx, ny) | distance to the nearest line in m, truncated at 2 |
| `centreline` | (nx, ny) | mask of the lane / lane-connector centre-lines |
| `centreline_dir` | (2, nx, ny) | direction of travel on `centreline` cells: a unit vector (the mean over the lines in the cell), (0, 0) where they cancel out |
| `vehicle` `pedestrian` `two_wheeler` | (nx, ny) | footprint coverage of **observed** boxes (at least one lidar or radar point) |
| `instance` | (nx, ny) int32 | index into `meta["boxes"]` of the box that covers the cell, -1 = none |
| `ignore` | (nx, ny) | boxes WITHOUT any lidar / radar point: neither object nor free space, leave out of every loss |
| `velocity` | (2, nx, ny) | world velocity (m/s, ego axes) of the covering box, radar-corrected; 0 elsewhere (the static world) |
| `flow_ego` | (2, nx, ny) | apparent velocity of a static world point from the ego motion, -(v + omega x r) |
| `flow` | (2, nx, ny) | `flow_ego + velocity`: the apparent motion in the camera frame; displacement over any dt = `flow * dt` |
| `flow_weight` | (nx, ny) | 1 = velocity known, 0 = unknown (ignored boxes, unknown velocity): the supervision mask |
| `flow_sigma` | (nx, ny) | std of the velocity in m/s: 0.1 in free space, 10 on every cell with `flow_weight` 0, always finite |
| `camera` | (nx, ny) | ground cells that project into the CAM_FRONT image |
| `occluded` | (nx, ny) | ground cells a box hides from the camera, by the box's real height (section 3a) |
| `radar` | (nx, ny) | cells inside the +-60 deg fan of one of the five radars |
| `undefined` | (nx, ny) | cells neither camera nor radar can see |

`meta`: `sample` / `scene` (tokens), `timestamp` (camera time, us), `annotation_timestamp` (us), `boxes_at` ("camera"), `T_e2g` (4x4 ego -> global at the camera time), `res`, `cam_file` (image path
relative to the dataroot), `K` (3x3 intrinsics), `T_c2e` (4x4 camera -> ego), `image_size`, `radar_poses` (five radar -> ego 4x4), `vectors` (exact polylines in the ego frame per `lane_divider` /
`road_divider` / `centreline`), `boxes` (every annotated box of the three classes, also those outside the grid, more than half of them: filter by `instance` or the corners). The "prev" / "next" metas of
a training sample add `dt` (s, signed) and `steps` ((dt, ego speed m/s, yaw rate rad/s) per 12 Hz camera frame).

One box of `meta["boxes"]`: `group` ("vehicle" / "pedestrian" / "two_wheeler"), `corners` (4x2 footprint, ego x, y), `center` (3, ego, at the camera time), `size` ([width, length, height] m), `yaw`
(rad relative to the ego heading, in [-pi, pi]), `vel` (2, world velocity m/s in ego axes), `vel_source` ("stationary" / "track" / "track+radar" / "radar" / "unknown"; "devkit" with `use_flow=False`),
`vel_flag` ("ok", "unobserved", "short_track", "stationary_but_moving", "radar_override", "radar_conflict"), `vel_sigma` (m/s; NaN when unknown), `vel_known` (False = unknown: `vel` is (0, 0), never use it
without this flag), `visibility` (nuScenes annotator bin "1" .. "4" = 0-40 / 40-60 / 60-80 / 80-100 %), `token` (annotation), `instance` (instance token), `stationary` (one fixed world pose, velocity
0), `observed` (has lidar / radar points), `world` (3, global position at the camera time), `world_ann` (at the annotation time), `shift` (2, ego axes, m: what moved the box from annotation to camera
time), `shift_s` (s), `num_lidar_pts` / `num_radar_pts` (nuScenes counts, information only), `share_camera` / `share_radar` / `share_fusion` (0 .. 1: the share of the footprint the camera sees / the
radar fan covers / both).

## 3a. Coverage: what the rig observes (D36 - D38)
The front rig is one camera and five radars. **Fusion area** = `camera` cells not `occluded`, inside the `radar` fan; outside it everything is **don't care on both sides** (training and evaluation).
`coverage.py` (numpy only, calibration + grid) holds the one definition. Public: `GtGenerator.coverage_for(g, kind)` (`map` / `objects` / `velocity`: the fusion area; `camera_aux`: camera cells not
hidden; `radar_aux`: the fan), `GtGenerator.radar_cells(gx, gy, poses)`, `GtGenerator.shares_in`. Inside the folder it is built from `fusion_cells(g)`, `camera_observed(g)`, `camera_cells(gx, gy, K,
T_c2e, size, z=...)`, `rig_radar_poses()` (the mean rig, for code without a calibration), `hidden_cells`, `footprint_share`.
- A box hides the ground behind it with its real volume (a low box casts a short shadow). Only observed boxes hide. Boxes stand on the ground plane z = 0 of the ego frame for this purpose; a point inside
  a box's footprint is never hidden by that box.
- **f = `share_fusion`**: the share of the footprint where the camera sees the object (at the bottom, middle or top of its height, not hidden by another box) AND the radar covers; samples outside the
  grid count as unobserved. Care f >= 0.5, partial 0.1 <= f < 0.5, don't care f < 0.1. A detection gets its f from the same function, `shares_in(g, corners, height or group=...)`; it counts every observed
  box of the frame as an occluder, also the one the detection is: pass a frame whose `meta["boxes"]` leaves a matched box out.
- Two static coverage masks without boxes (`rig_masks.py`, from the calibration): `GtGenerator.mask_for("bev_fusion")` (front camera + radars on the grid above, (120, 104)) and
  `GtGenerator.mask_for("lane_center")` (six cameras + radars, (200, 200), lane frame x right, y forward) -> `{"camera", "radar", "fusion"}`; another name is an error. `python -m GtGenerator masks out.png`.

## 3b. The lane GT (`lane_gt.py`; own signature and stores)
Read with `GtGenerator.load_lane(token, rig=...)`; `GtGenerator.lane_tokens(rig=...)` are the stored tokens, `GtGenerator.lane_signature(rig=...)` is the key of the store. Made from the nuScenes map and the ego pose at the front-camera time; the tiers and the visibility also read the CAM_FRONT ego poses around the keyframe and the annotated boxes. Two **rigs**, one store
each; geometry, ids, events and tiers are the same, only frame, window and cameras differ:
| rig | frame | window (`g["half"]`, `g["window"]`) | `vis` / `occ` |
|---|---|---|---|
| `lane_center` (Spurmitte, default) | "lane": x right, y forward | the square \|x\|, \|y\| <= 50 m | one of the six cameras |
| `bev_fusion` | "ego": x forward, y left | the BEV grid x 0 .. 60, y -26 .. 26 m (`window`, key `rig`; `half` 60 is only a bound) | CAM_FRONT |

A graph is a dict with `frame` (ego rear axle at the camera time; `lane_graph.in_frame(g, "ego")` turns it), `half`, `location`, `sample`, `signature`, `params` and:
- `pieces`: lanes and connectors joined while the join is 1 in, 1 out (a new id only at a split or a merge), cut to the window: `id`, `piece`, `part` (a piece that comes back into the window is "<id>#1",
  a number that depends on the window: use `piece` for anything stable across frames), `kind` (lane / connector / mixed), `pts` (n, 2) float32 every 0.5 m in travel order, `start_open` / `end_open` (cut
  by the window), `length`, `segments` (`[{token, kind, i0, i1, in_intersection}]` per map piece), `succ` / `pred`, `left` / `right` / `opposite` (ids), `road`, `lane_count`, `lane_index` (0 = leftmost);
  per point `tier`, `tier_src`, `vis`, `occ`.
- `events`: `type` split / merge / cross / stop / exit / start / end / window, `pos`, `in` / `out` (ids), `dirs_in` / `dirs_out` (unit vectors in the same order), `partial` (a branch is outside the window),
  `tier`, `vis`; stop / exit: `on`, `s` (metres along the part), `n_succ` / `n_pred`; window: `io`. A stop is the end of a LANE whose successor connector lies inside an intersection, an exit the start of a
  LANE whose predecessor connector does (`stop_rule`); both can lie inside a joined piece (then `out` of the stop, `in` of the exit, is that piece). Every stop / exit names its in and out.
- `roads` (derived from the lanes, nuScenes has no road graph): `id`, `kind` (road / intersection), `lane_max`, `pieces`, `succ`, `opposite`.
- `lines`: `lane_divider`, `road_divider`, `boundary` (outline of the drivable area): `{pts, start_open, end_open, corners}`, a corner `{pos, angle, idx, dir_in, dir_out}`.
- `areas`: `road` and `intersection` polygons `{id, exterior, holes, area}` (the id of an intersection is its node in `roads`, None without a lane connector), not dilated. `lane_raster.area_mask(g, kind,
  border_m)` draws one into the BEV grid (a cell is set when its centre is inside; rig `bev_fusion` only, the other window does not reach 60 m).
- `ego_path` (`pts`, `back_m`, `ahead_m`), `tier_info` (metres per tier, visible metres, hiding boxes, extension).

**Tiers** (labels only): 1 = the ego lane on the path the car really drives, 25 m back to 55 m ahead (**NON-CAUSAL**: never a model input), extended along the map where the driven path is short
(`tier_src` 2; 1 = driven, 0 = not tier 1; the extension labels whole pieces, so it can reach farther); 2 = left / right neighbours within 10 m of tier 1; 3 = the rest. **`vis`** = a camera of the rig sees
the ground point and no observed vehicle / pedestrian / two-wheeler box hides it; `occ` = in view but hidden; both False = outside every view (as within about 4.5 m of the car, so a consumer must not
read that as "hidden"). Cameras only; buildings are not boxes, so a point behind one counts as visible; boxes are at the annotation time. Parameters (`lane_gt.DEFAULTS`; another value = another signature and
store) include `boundary_layers`, `corner_deg`, `stop_rule` and the tier distances.

## 4. Rules and recipes
- **Signature = identity of the GT.** An edit of a hashed file (README, versions) makes every stored GT unreadable for the new code until a version row and a rebuild exist: never edit one while a job
  runs, tell the owner of the training code before, never compare models on different signatures, and put the signature (and VAL / TEST, the denominator) next to every number. `light_nusc.py` is not hashed
  although it decides which records the GT sees: after an edit run the self-test. A process keeps the signature it computed first: restart a reader after an edit.
- **Boxes are at the camera time**: a moving box of known velocity is moved by `v * (t_camera - t_annotation)` (about -36 ms; stationary and unknown-velocity boxes stay);
  `boxes_at="annotation"` restores the annotation positions (another signature). A dt is a model input only to predict to ANOTHER time.
- **Stationary means stationary**: a parked car has ONE world pose and velocity 0 in all keyframes; a map correction would be per element in world coordinates, never per frame.
- **Where a loss or score counts**: `coverage_for` (section 3a). Objects only where `observed`, leave `ignore` out, velocity only where `flow_weight > 0`.
- **"Moving" has one definition**: known velocity above 1 m/s - cells `(np.hypot(*g["velocity"]) > 1) & (g["flow_weight"] > 0)`, boxes `b["vel_known"] and np.hypot(*b["vel"]) > 1`. `stationary` is
  something else (a creeping box is neither). About 80 % of the supervised object cells are static: weight moving and static cells separately and report speed errors for both, next to the zero predictor.
- **Map holes**: the nuScenes map misses some new roads (about 0.3 % of keyframes, Boston Pier 4 construction): `g["drivable_area"][2:8, 50:54].mean() < 0.5` detects them; leave them out of a map loss.
- **Limits**: boxes exist at 2 Hz (yaw is not extrapolated); the camera offset was measured for CAM_FRONT only; Boston map lines are 0.3 - 0.7 m off in places; about a quarter of the boxes stand more
  than 0.5 m above or below the ground plane (a 2D map: cars on elevated roads lie outside the drivable area); the occlusion mask is 2D; radar gives only the radial velocity; nothing is validated
  against lidar.

Recipes:
```python
g = GtGenerator.load(tok)
moving = (np.hypot(*g["velocity"]) > 1.0) & (g["flow_weight"] > 0)
sup = (g["vehicle"] > 0) & (g["flow_weight"] > 0) & (g["ignore"] == 0)         # supervised vehicle cells
free = (g["drivable_area"] > 0.5) & (g["ignore"] == 0) & (g["undefined"] == 0)
b = g["meta"]["boxes"][int(g["instance"][40, 52])]                              # the box of cell (40, 52); instance -1 = none
s = gen.training_sample(tok)       # needs the generator; None at the first / last keyframe; {"prev", "cur", "next"}, each in its own ego frame
dt = s["next"]["meta"]["dt"]       # about +0.5 s; s["cur"]["flow"] * dt = first-order displacement of every cell; box["instance"] links the objects
```
The velocity / flow estimator (`estimate_flow(tracks, ego, t, radar, config)` in `flow.py`, dataset independent; inside the folder, not part of the interface) takes the ego velocity from the poses, one pose and v = 0 for a stationary track, else a
line fit over +-1.2 s, then a robust radar correction of the radial component, and flags what disagrees.
