# bev_fusion - BEV Fusion

A camera + radar bird's-eye-view perception system on nuScenes: map layers, objects and motion from the front camera and five radars, with a
memory carried along a scene. What it must satisfy: `PROBLEM.md` (the owner's requirements); the modules and their rules: `DESIGN.md`;
the owner's decisions: `DECISIONS.md`; where things stand: `HANDOVER.md`; the champion (the owner's open choice) and the measured candidates:
`../../champions/bev_fusion/`.

| folder | what |
|---|---|
| `net/` | the model in three layers, `prep/` < `model/` < `loss/`, and `chain.py` on top: a run's config, model, weights, forward and loss for the trainer and the evaluation alike ([`net/README.md`](net/README.md)) |
| `training/` | `train.py` on `trainkit`, `freeze.py`, `build_image_cache.py`, `configs/base.json` (the config of the candidate `t21_ctl`), the drawing `architecture.*` |
| `evaluation/` | `bev_eval.py` (the adapter of `Evaluation`), `bev_adapter.py` (source, subject), `bev_task.py` (metrics, views, the score), `metrics.py`, views and further metric families; it reads the network through `net/chain.py`, never through `training/` |
| `tools/import_ckpt.py` | the step `import`: a measurement of an existing network takes its checkpoint |
| `splits/` | the scene lists (small, large, xl) |
| `tests/` | one test file per module (`test_<module>.py`), the layering and import rules (`test_net_layering.py`), `net/chain.py` (`test_chain.py`), the layer table, the splits, the evaluation adapter contract, the golden training run, the end-to-end chain; `support/` their synthetic world and batches |

Modules name each other by their package path from this folder (`from net.model.model import ModelConfig`); no file edits `sys.path`. `research` puts this folder,
`autoresearch/` and the repository top on `PYTHONPATH` for steps and tests; by hand, from this folder:
`PYTHONPATH=.:../../autoresearch:../.. ~/miniconda3/envs/gcar/bin/python tests/test_chain.py` (the same for `training/train.py <config>`, the cache builders
`net/prep/build_feature_cache.py`, `cache_bevfusion_features.py`, `gt_cache_build.py` and the tools `evaluation/bev_timing.py RUN`, `bev_modules.py RUN`).

## Data and runs (not in git; where the machine file says)
The code works in `<data root>/bev_fusion/` (`net/netpath.SITE`), so the configs name files relative to it: `data/cache/gt` (the
ground-truth cache of GtGenerator, front v13), `data/cache/cam_feats` and `cam_feats_bevfusion` (camera features), `data/cache/radar`,
`data/cache/img_front`, `data/cache/annotation_pts.npz` (lidar and radar point counts of the boxes), `models/pretrained/` (DualBEV, needed
at every model build; BEVFusion camera-only-seg, only to rebuild its cache) and `runs/` (a link to `<runs root>/bev_fusion`, where the
imported candidates `t20_base_s0`, `t21_ctl_s0`, `t21_ctl_s1` are). Raw nuScenes: `~/data/nuscenes_full`. The `gt` field of `splits/*.json`
names an older ground-truth version than the cache's.

## Use (from `autoresearch/`, under WSL)
```
python3 -m research new bev_fusion <slug> "<title>"     # arms change keys of training/configs/base.json (nested ones key by key)
python3 -m research start bev_fusion E<n>               # steps train + evaluate; measuring an existing network: "steps": ["import", "evaluate"] and the setting "_import": "runs/<run>"
python3 -m research judge bev_fusion E<n> --write
```
The score `board.board7` is the mean of six map rows in the field of view and the object mean average precision (`evaluation/bev_task.py`),
on the frames `problem.json` names (VAL, evaluated warm, D70). A repeated evaluation on the GPU differs slightly (`anchor_tolerance`); the
CPU is deterministic. Tests: `python3 -m research tests bev_fusion` (CPU, 2 threads, about half an hour; every file of `tests/` is registered in `problem.json`).
What they need beside the code: nuScenes mini under `~/data/nuscenes` (`test_model`, `test_augment`, `test_nuscenes_dataset`, `test_camera_encoder`,
`test_images`, `test_feature_cache`, `test_gt_cache_build`, `test_build_feature_cache`, `test_end_to_end`), the pretrained weights of the site
(`models/pretrained/`: the same and `test_lane_branch`), and the site's mini feature cache `data/cache/cam_feats_mini` with the BEVFusion weights
(`test_cache_bevfusion_features`); `python tests/test_evaluation_api.py --real` (not registered: a finished run, 18 GB, 5 minutes) compares the class
interface with the engine's own run. The ground-truth cache `data/cache/gt` is filled by the dataset on first use, or ahead of a training by
`net/prep/gt_cache_build.py <nuScenes root> <version> data/cache/gt --split train|val`.
A stored config that names an option which left the code loads when the option is at its off value and is refused otherwise (`net/chain.py` `REMOVED`).
