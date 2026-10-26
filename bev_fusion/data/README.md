# data - derived data of bev_fusion (not in git)

Everything here is derived from the raw nuScenes data (`D:\Auto-research\NuScenes`, WSL views `~/data/nuscenes_full`,
`~/data/nuscenes`; `../../datasets/NuScenes/README.md`) and the pre-trained checkpoint (`../models/README.md`) by tracked
scripts, so it can be deleted and rebuilt (rule D11 of the lane project: a new derived artifact gets its row here in the same
step). `.npy` / `.npz` / `.pt` files are ignored by git globally. Folders: `cache/` (this file's table), `../runs/` (training runs).

| artifact | made by | inputs | status |
|---|---|---|---|
| `cache/cam_feats_<set>/<scene>.npy` + `.index.npz` - outputs of the frozen camera heads (118 depth-probability + 80 context channels at 16 x 44, float16, 0.28 MB per frame); the frames the dataset uses (`n_cam`, `cam_stride`) | `python training/feature_cache.py build <dataroot> <version> <root> [--split train\|val] [--device cuda]` (resumable per scene) | CAM_FRONT images, calibration, `models/pretrained/dualbev-r50-cbgs.pth` | scripted; mini: `cam_feats_mini` (first 6 samples); full: not built yet (about a third of the 98 k front frames, roughly 9 GB; the GPU build takes about 30 min, needs an idle GPU) |
| `cache/gt/<token>_<res>_<ss>.npz` - GT of one keyframe at k - 1, k, k + 1 on the grid (`FrontRadarDataset(cache_dir=...)`, float16) | written on first use by `training/nuscenes_dataset.py` (`NuScenesGT.training_sample`) | nuScenes map expansion, annotations, poses | scripted (lazy) |
| `../runs/<name>/` - checkpoints (trainable weights, optimizer, EMA, RNG; the frozen camera heads are not stored), `log.jsonl`, `config.json` | `python training/train.py <config.json>` | the two caches above or the raw data | scripted (re-train; GPU training agrees within the noise band, not bit-exactly) |

Size guide (full nuScenes, train + val scenes of parts 1-6): feature cache about 9 GB with `n_cam = 3`, `cam_stride = 2`
(all 98 k front frames would be 27 GB); GT cache about 0.5 MB per keyframe.
