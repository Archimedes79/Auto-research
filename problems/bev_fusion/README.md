# bev_fusion - BEV Fusion

A camera + radar bird's-eye-view perception system on nuScenes: map layers, objects and motion from the front camera and five radars, with a
memory carried along a scene (D70). What it must satisfy: `PROBLEM.md` (the owner's requirements); the modules and their rules: `DESIGN.md`;
the owner's decisions: `DECISIONS.md`; where things stand: `HANDOVER.md`. Champion: `t21_ctl` (the t20 network continued 1500 steps from
`t20_base_s0`), see `champions/bev_fusion/`.

| folder | what |
|---|---|
| `net/prep/` | data: the nuScenes dataset with front camera and radar, the feature caches (DualBEV and BEVFusion camera features of frozen pre-trained networks), ground-truth targets, splits |
| `net/model/` | the network: camera lift, radar encoder, temporal fusion, the DualBEV trunk, map / object / velocity heads, the lane branch, the motion block |
| `net/loss/` | the losses of the heads |
| `training/` | `train.py` on `trainkit`, `freeze.py`, `probe.py`, `gt_scenes.py`, `build_image_cache.py`, `configs/base.json` (the champion's own config), the drawing |
| `evaluation/` | `bev_eval.py` (the adapter of `autoresearch/Evaluation`), `bev_adapter.py` (source, subject, the chain), `bev_task.py` (metrics, views, the score board7), `metrics.py`, views and the extra metric families |
| `tools/import_ckpt.py` | the step `import`: a measurement of an existing network takes its checkpoint |
| `splits/` | the scene lists (small, large, xl) |
| `tests/` | layer tests, the trainkit equivalence, the end-to-end chain |

## Data and runs (not in git; where the machine file says)
The code works in `<data root>/bev_fusion/` (`net/netpath.SITE`): its configs name `data/cache/gt` (the ground-truth cache of GtGenerator v13),
`data/cache/cam_feats` and `cam_feats_bevfusion` (camera features), `data/cache/radar`, `data/cache/img_front`, `models/pretrained/`
(DualBEV, needed at every model build; BEVFusion camera-only-seg, only to rebuild its cache) and `runs/` (a link to `<runs root>/bev_fusion`,
where the champion's runs `t20_base_s0`, `t21_ctl_s0`, `t21_ctl_s1` keep their names). Raw nuScenes: `~/data/nuscenes_full`.

## Use
```
python -m research new bev_fusion <slug> "<title>"     # arms change keys of training/configs/base.json (nested ones key by key)
python -m research start bev_fusion E<n>               # steps train + evaluate; a measurement of an existing network: "steps": ["import", "evaluate"], "_import": "runs/<run>"
python -m research judge bev_fusion E<n> --write
```
The score `board.board7` is the mean of the six map rows in the field of view and the object mAP, from the counts of one paired bootstrap; the
evaluation runs warm (D70) on 1000 frames spread over the 82 VAL scenes. On the GPU the numbers of a repeated evaluation differ by up to 3e-3
(`anchor_tolerance`); the CPU is deterministic. Older numbers (`keep3` board7 0.4549, cold, ground truth v8) are history in the tag `v1`.
