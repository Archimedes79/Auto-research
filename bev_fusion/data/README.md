# data - derived data of bev_fusion (not in git)

Everything here is derived from the raw nuScenes data (`D:\Auto-research\NuScenes`, WSL views `~/data/nuscenes_full`,
`~/data/nuscenes`; `../../datasets/NuScenes/README.md`) and the pre-trained checkpoint (`../models/README.md`) by tracked
scripts, so it can be deleted and rebuilt (rule D11 of the lane project: a new derived artifact gets its row here in the same
step). `.npy` / `.npz` / `.pt` files are ignored by git globally. Folders: `cache/` (this file's table), `../runs/` (training runs).

| artifact | made by | inputs | status |
|---|---|---|---|
| `cache/cam_feats_<set>/<scene>.npy` + `.index.npz` - outputs of the frozen camera heads (118 depth-probability + 80 context channels at 16 x 44, float16, 0.28 MB per frame); the frames the dataset uses (`n_cam`, `cam_stride`) | `python training/feature_cache.py build <dataroot> <version> <root> [--split train\|val] [--device cuda]` (resumable per scene); the whole set by `temp/setup/build_full_cache.sh` | CAM_FRONT images, calibration, `models/pretrained/dualbev-r50-cbgs.pth` | scripted; mini: `cam_feats_mini` (first 6 samples); full: `cam_feats` built 2026-10-26 on the shared GPU at 11-16 frames/s: 58,101 frames, 16 GB, 510 scenes (train 428 scenes / 16,289 samples, val 82 scenes / 3,109 samples: the nuScenes parts downloaded so far), every sample finds its 3 frames |
| `cache/gt/<token>_<res>_<ss>_<signature>.npz` - GT of one sample = keyframes k - 1, k, k + 1 on the grid with the ego steps (`FrontRadarDataset(cache_dir=...)`, float16); the key contains the signature of the generator source `gt_bev.py`, so an edit of it means a new cache | written on first use by `training/nuscenes_dataset.py` (cold: 12 s per sample before the 2026-10-26 speed-up, about 1.5 s after); prefilled before training by `python training/gt_cache_build.py ...` / `temp/setup/build_full_gt_cache.sh` (4 workers, 0.5-0.6 s per sample with the keyframe memo) | nuScenes map expansion, annotations, poses | scripted (prefilled 2026-10-26: val 3,109 samples in 480 s, train 16,289 samples in 2,236 s wall) |
| `../runs/<name>/` - checkpoints (trainable weights, optimizer, EMA, RNG; the frozen camera heads are not stored), `log.jsonl`, `config.json` | `python training/train.py <config.json>` | the two caches above or the raw data | scripted (re-train; GPU training agrees within the noise band, not bit-exactly) |

Size (the nuScenes parts downloaded so far, `n_cam = 3`, `cam_stride = 2`): feature cache 16 GB (58 k frames); GT cache 2.6 GB (19,398 samples, about 0.13 MB per sample).
