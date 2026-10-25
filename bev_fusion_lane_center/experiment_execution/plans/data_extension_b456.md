# Data extension b456: the 255 nuScenes scenes of parts 4-6 (plan of 2026-10-25, decision D1)

## What is on disk
- nuScenes trainval: 850 scenes. Camera keyframes of parts 1-6 (510 scenes) were already on disk; radar only of parts 1-3 (255 scenes), and the project uses exactly those 255 (TRAIN 212 / VAL 22 / TEST 21 scenes, 8,414 / 873 / 833 frames).
- D1 (approved): radar parts 4-6 from the public bucket `motional-nuscenes` (AWS Open Data, no login): `v1.0-trainval0{4,5,6}_blobs_radar.tgz`, 215 + 215 + 202 MB, downloaded into `~/downloads/nuscenes_radar_456/`, each part checked against the S3 ETag of the bucket listing. Status 2026-10-25 12:57: all three parts verified (ETag OK), unpacked into `~/data/nuscenes_full` with `~/extract_blob.sh` (path check): radar keyframes 20,418 = camera keyframes 20,418, so all 510 scenes now have camera and radar; radar sweeps 112,774. The archives stay in `~/downloads/nuscenes_radar_456/` (632 MB) until the extraction is checked.
- The new 255 scenes:

| official split | scenes | frames | locations |
|---|---|---|---|
| train | 216 | 8,731 | boston-seaport 137, singapore-onenorth 40, singapore-queenstown 20, **singapore-hollandvillage 19** (a location the project has never seen) |
| val | 39 | 1,567 | boston-seaport 36, singapore-onenorth 3 |

## Splits (decision D7 for Mirko: a change of the data and the protocol)
Recommendation: **keep TRAIN / VAL / TEST exactly as they are** (18 loops of history stay comparable) and add:
- **TRAIN2** = TRAIN + the 216 new official-train scenes (17,145 frames, about twice the data);
- **CONFIRM** = the 39 new official-val scenes (1,567 frames): never used for tuning or selection, only to confirm a champion candidate before its one TEST check (the fresh holdout that VAL no longer is);
- option: hold the 19 hollandvillage scenes out of TRAIN2 as an **unseen-location** test (about 750 frames), since no location of TEST is new to the model.

## Pipeline for the new scenes (set `b456`, written to new folders only; nothing of b123 is touched)

| step | what | tool | output | check (before the next step) | time |
|---|---|---|---|---|---|
| 1 | unpack the radar parts (path check against `..`) | `~/extract_blob.sh <tgz>` | `~/data/nuscenes_full/samples/RADAR_*`, `sweeps/RADAR_*` | radar keyframes for 510 scenes | 5 min |
| 2 | token lists of the new frames (official train -> train, official val -> val), ordered by scene and time like b123 | one-off script in `temp/loop19/` | `~/exp_gcar_b456/feats/{train,val}_tokens.json` | 8,731 / 1,567 tokens, scenes disjoint from b123 | 5 min |
| 3 | extraction check: re-extract 50 b123 frames with the same code and the **b123 PCA basis** and compare with the stored features | `gcar_tools.py extract --check` | - | correlation >= 0.999 (rasteriser noise level) | 5 min |
| 4 | GaussianCaR features `dec32`, `logit7` (and `fused32`, optional, unused by the pipeline) of the new frames, b123 PCA basis and orientation | `gcar_tools.py prepare / extract --split b456` (2 processes) | `~/exp_gcar_b456/feats/*.npy` (about 30 GB) | row count, no NaN, value range like b123 | about 2.5 GPU hours |
| 5 | ground truth per frame: graph (`make_graph_gt.py`), exact distance (`make_dist_gt.py`), direction (`make_dir_gt.py`), poses and previous frames (`make_poses.py`) | prep scripts with the set as a parameter | `~/exp_gcar_b456/{graph_gt,dir,poses}_*`, `gt_dist_exact` | regenerate 20 b123 frames: identical to the stored ones | about 1 CPU hour |
| 6 | route and map: SD-map simulation (`sd_precompute.py`), route ground truth (`make_route_gt.py`, `make_route_heat.py`), map priors exist for all four locations | prep scripts | `~/exp_gcar_b456/{sd,route_gt,route_pts,route_heat}_*` | same 20-frame regeneration check | about 1 CPU hour |
| 7 | loader: data sets as a list of roots (b123, b456) instead of one hard-coded `exp_gcar_b123`; split names TRAIN2 / CONFIRM; `evaluate.py --split confirm` | code change of the core bundle (roadmap step `lcr.paths`, `ar-code-reviewer`) | - | golden numbers on VAL unchanged; b456 rows load | 0.5 day |
| 8 | optional, later: perturbed features (failure series) of the new frames | `gcar_pert.py` | `~/exp_gcar_b456/pert/` (about 37 GB) | - | about 17 GPU hours (7 failure types); only if a loop needs more failure data |

Steps 1-4 write only new files and need no change of shared code except a `b456` entry in `gcar_tools.py` (a prep script no experiment imports), so they can run right away on the idle GPU. Steps 5-7 come with roadmap M0 / M1 (golden numbers first), because they touch the shared data code.

## What it enables (hypothesis card for cycle 19)
"Twice the training scenes closes part of the coverage gap": retrain the champion recipe on TRAIN2 (equal steps and 2x steps) against the same recipe on TRAIN, both seeds; evaluate on VAL (history) and CONFIRM (fresh); prediction: lanes found and R@1 far up by more than the noise band, topo F1 up; the control E94 gets the same data for a fair comparison.
