# Data

## In this folder
- Raw nuScenes is on D: and is described once for all problems in `../../datasets/NuScenes/README.md`; this file lists only this problem's own data.
- `nuscenes_additions/` (not in git, since 2026-10-25): this problem's derived nuScenes files, `nuscenes_additions/lcr_crn/` (CRN info files `nuscenes_infos_{train,val}{,_crn}.pkl`, `nuscenes_additions/lcr_crn/radar_bev_filter/`, `nuscenes_additions/lcr_crn/radar_pv_filter/`; 70,844 files) and `nuscenes_additions/lcr_crn_mini/` (the same for v1.0-mini), `nuscenes_additions/manifests/` (file lists of the copy). The WSL paths the code uses (`~/data/nuscenes_full/<file>`, `~/data/nuscenes/<file>`) are links to here.
- No nuscenes folder any more (removed 2026-10-25): the Windows copy of nuScenes mini duplicated `~/data/nuscenes` in the WSL home, which the code reads.
- `splits.json` - TRAIN / VAL / TEST scene lists of the full runs (scene-disjoint).
- The raw nuScenes archives (`v1.0-mini.tgz`, map expansion, `v1.0-trainval_meta.tgz`, radar parts 1-3) were unpacked into the WSL home (see below); the archives were removed on 2026-10-25 (purged from _to_delete); the camera archives were not kept. Radar parts 4-6 were downloaded on 2026-10-25 (decision D1; public bucket `motional-nuscenes`, AWS Open Data, no login; verified by S3 ETag; archives in `~/downloads/nuscenes_radar_456/`). nuScenes is licensed for non-commercial use; re-download only with the user's OK.

## Not in this folder (and why)
The big derived arrays live in the **WSL home** (`\\wsl$\Ubuntu-22.04\home\ml`, ext4): reading them over `/mnt/c` is about an order of magnitude slower. Registry (sizes of 2026-10-24):

| path in WSL | size | content | how it is made |
|---|---|---|---|
| `~/data/nuscenes_full` | 28 GB | nuScenes parts 1-6: camera keyframes and radar (samples + sweeps) of 510 scenes (radar of parts 4-6 since 2026-10-25), metadata, infos of parts 1-3 | public bucket `motional-nuscenes` (archives of parts 1-3 removed after unpacking), `training/data/prep/crn_make_infos.py`; the project uses 255 scenes, the other 255 are planned in `../experiment_execution/plans/data_extension_b456.md` |
| `~/data/nuscenes` | 3 GB | mini copy for the env | |
| `~/exp_gcar_b123/feats` | 54 GB | cached GaussianCaR features (dec32, fused32, logit7) per frame, TRAIN / VAL / TEST | `training/data/prep/gcar_pipeline1.sh`, `gcar_pipeline2.sh`, `extract_features.py` |
| `~/exp_gcar_b123/pert` | 37 GB | features under sensor failures (dark, blur, cameras off, ...) | `training/data/prep/gcar_pert.py` |
| `~/exp_gcar_b123/p17` | 39 GB | failure series of loop 17 (clean / failure half per scene, out-of-fold heads) | `p17_series.py` (to be restored, see HANDOVER) |
| `~/exp_gcar_b123/p8, p10, p14` | 11 / 5.4 / 6.4 GB | token caches and chain data of loops 8-16 | `p8_*`, `p10_*`, `p14_chain_data.py`, `p16_chain_data.py` |
| `~/exp_gcar_b123/*.npy, *.pkl, *.npz` | ~4 GB | ground-truth graphs (`graph_gt_{train,val}.pkl`), direction / node / route maps, poses, SD-map priors and registrations | `training/data/prep/make_*.py`, `prior_precompute.py`, `sd_precompute.py` |
| `~/exp_crn_b123`, `~/exp_crn_mini`, `~/exp_gcar_mini` | 12 / 5.7 GB / small | CRN features (loop 1), dense ground-truth points `*_gt_pts_exact.npy`, mini caches | `training/data/prep/crn_*` |
| `~/work/CRN`, `gaussiancar`, `bevfusion`, `gaussiancar_ckpt` | 2.4 GB | clones of the foundation code and weights | `../literature/code/README.md` |
| `~/miniconda3/envs/gcar` | | environment, package list in `../tools/env/environment_gcar.txt` / `.yml` | `tools/env/setup/setup_gcar_env.sh` |

Rebuilding a cache from nothing: download nuScenes (ask first), run the `data_prep` scripts in the order of the loop-1/2 notes (`experiment_execution/history/`), then the feature extraction (long on one RTX 3060; timings are in the loop-2 notes). Do not delete these caches to "clean up".
