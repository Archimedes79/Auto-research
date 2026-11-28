# bev_fusion - BEV Fusion

A camera + radar bird's-eye-view perception system on nuScenes: map layers, objects and motion from the front camera and five radars, with a
memory carried along a scene. What it must satisfy: `PROBLEM.md` (the owner's requirements); the modules and their rules: `DESIGN.md`;
the owner's decisions: `DECISIONS.md`; where things stand: `HANDOVER.md`; the champion (the owner's open choice) and the measured candidates:
`../../champions/bev_fusion/`.

| folder | what |
|---|---|
| `net/` | the model in three layers, `prep/` < `model/` < `loss/`, and `chain.py` on top: a run's config, model, weights, forward and loss for the trainer and the evaluation alike ([`net/README.md`](net/README.md)) |
| `training/` | `train.py` on `trainkit`, `freeze.py`, `build_image_cache.py`, `configs/base.json` (the config of the candidate `t21_ctl`; its scenes by name: `"split": "large"`), the drawing `architecture.*` |
| `evaluation/` | `bev_eval.py` (the adapter of `Evaluation`), `bev_adapter.py` (source, subject), `bev_task.py` (metrics, views, the score), `metrics.py`, views and further metric families; it reads the network through `net/chain.py`, never through `training/` |
| `tools/import_ckpt.py` | the step `import`: a measurement of an existing network takes its checkpoint |
| `splits/` | the scene lists (small, large, xl) |
| `tests/` | one test file per module (`test_<module>.py`), the layering and import rules (`test_net_layering.py`), `net/chain.py` (`test_chain.py`), the layer table, the splits, the evaluation adapter contract, the golden training run, the end-to-end chain; `support/` their synthetic world and batches |

The problem is the package `bev_fusion`: a module names another absolutely (`from bev_fusion.net.model.model import ModelConfig`), every code folder holds an empty
`__init__.py`, no file edits `sys.path`, and everything is started as a module with `python -P -m`. `research` gives steps and tests the module path (`problems/`,
`autoresearch/` and the repository top, of the run's snapshot); nothing is installed. By hand, from `autoresearch/`:
`python3 -m research py bev_fusion -m bev_fusion.tests.test_chain` (the same for `-m bev_fusion.training.train <config>`, the cache builders
`bev_fusion.net.prep.build_feature_cache`, `cache_bevfusion_features`, `gt_cache_build` and the tools `bev_fusion.evaluation.bev_timing RUN`, `bev_modules RUN`).

## Data and runs (not in git; where the machine file says)
The code works in `<data root>/bev_fusion/` (`net/netpath.SITE`), so the configs name files relative to it: `data/cache/gt` (the
ground-truth cache of GtGenerator, front v13), `data/cache/cam_feats` and `cam_feats_bevfusion` (camera features), `data/cache/radar`,
`data/cache/img_front`, `data/cache/annotation_pts.npz` (lidar and radar point counts of the boxes), `models/pretrained/` (DualBEV, needed
at every model build; BEVFusion camera-only-seg, only to rebuild its cache) and `runs/` (a link to `<runs root>/bev_fusion`, where the
imported candidates `t20_base_s0`, `t21_ctl_s0`, `t21_ctl_s1` are). Raw nuScenes: the views `<data root>/nuscenes` and `nuscenes_mini` (`net/netpath.nuscenes_root`; the stored configs name the older link `~/data/nuscenes_full` to the same folder). The `gt` field of `splits/*.json`
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
What they need beside the code: nuScenes mini (`<data root>/nuscenes_mini`; `test_model`, `test_augment`, `test_nuscenes_dataset`, `test_camera_encoder`,
`test_images`, `test_feature_cache`, `test_gt_cache_build`, `test_build_feature_cache`, `test_end_to_end`), the pretrained weights of the site
(`models/pretrained/`: the same and `test_lane_branch`), and the site's mini feature cache `data/cache/cam_feats_mini` with the BEVFusion weights
(`test_cache_bevfusion_features`); `-m bev_fusion.tests.test_evaluation_api --real` (not registered: a finished run, 18 GB, 5 minutes) compares the class interface
with the engine's own run. The ground-truth cache `data/cache/gt` is filled by the dataset on first use, or ahead of a training by
`-m bev_fusion.net.prep.gt_cache_build <nuScenes root> <version> data/cache/gt --split train|val`.
A stored config that names an option which left the code loads when the option is at its off value and is refused otherwise (`net/chain.py` `REMOVED`).
Scenes: the base config names them by `"split": "large"` and keeps `train_scenes` / `val_scenes` empty (empty = the named split). A card that wants its own
lists sets them together with `"split": null`; a split next to a non-empty list is refused (neither wins), and so is `split: null` without a list.
