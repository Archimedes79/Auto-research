# lane_center - Spurmitte

A lane-centre-line graph from camera + radar (GaussianCaR features), end to end differentiable, planner-ready, good and fast. The goal and
its metrics: `PROBLEM.md`; the modules and the rules they obey: `DESIGN.md`; the owner's decisions: `DECISIONS.md`; where things stand:
`HANDOVER.md`; the champion (the two-stage token head on the lane head B') and its numbers: `../../champions/lane_center/`.

| folder | what |
|---|---|
| `net/prep/` | data: the token cache (`lane_tokcache.py`) and its loader (`lane_ptok_data.py`), the frames and the graph ground truth (`frames.py`, `lane_gt_data.py`, `lane_tok_data.py`), targets |
| `net/prep/foundation/` | the GaussianCaR features made again from raw nuScenes (`gcar_features.py`) and a check against the stored ones (`check_features.py`); its `README.md` names what must exist outside the repository |
| `net/model/` | the networks: the token head (`jtoken.py`, the pair geometry `jtok.py`), the lane head B' (`qgraph.py` and its parts), the soft read-out pieces |
| `net/loss/` | the token loss (`loss_jtoken.py`) and the losses of B' |
| `training/` | `train/lane_train.py` (the two stages from a run's config: its blocks `stage1` / `stage2` are the trainers' options, key by key; test `net/tests/test_lane_train.py`), `train/jtoken_train.py` (stage 1), `train/jlive_train.py` (stage 2, with B' live in the loop: `train/lane_live_tok.py`), `trainer.py`, `bflc_paths.py` (where code, data and runs are), `configs/base.json`, the drawing `architecture.*` |
| `evaluation/` | `lane_predict.py` (the step `predict`: the live dump), `adapter.py` (the evaluation step: decode and score through `Evaluation`), `jtoken_eval.py` (dumps, decodes, diagnostics), `jtiming.py` (the timing protocol), `qgraph_eval.py` + `sparse_lanes/` + `lane_task/` (the scorer and its metrics) |
| `tools/viz/` | pictures of graphs over BEV and camera |
| `data/splits.json` | the splits by row: TRAIN, VAL and TEST (the holdout) |
| `experiments/` | one folder per experiment |

## Data and runs (not in git; where the machine file says)
- `<data root>/lane_center/b123/`: the GaussianCaR features (`feats/{train,val}_{dec32,logit7}.npy`, tokens), the graph ground truth the
  headline numbers are scored against (`graph_gt_{train,val}.pkl`), poses (the scene of a frame). The features are made again by one
  command (`net/prep/foundation/README.md`: the GaussianCaR clone at a named commit, its weights by hash, raw nuScenes, the environment). A
  new extraction is not bit-identical, the rasteriser is not deterministic: measured on 44 frames (2026-11-28), dec32 differs from the
  stored arrays by 0.018 - 0.031 in the mean (mean absolute value 0.8; two new extractions differ by 0.016 - 0.020), correlation per frame
  0.993 - 0.999 in the median. So these files are kept.
- `<data root>/groundtruth/lane/9ad1d4ce30/`: the lane ground truth of GtGenerator; `<data root>/nuscenes/`: raw nuScenes (cameras for
  pictures).
- `<runs root>/lane_center/`: the runs of the experiments, the imported networks of the champion's chain (`e372_s4_s*`, `e373_cb_s*`,
  `e292_a_s*` = B', `e273_p_s*`, `e271_a_s*`), the token cache `e365_cache/`, the label tables `e357_cache/`, `e365_planb/`,
  `refiner_stats_dec32_logit7.npz`.

## Use (from `autoresearch/`, under WSL)
```
python3 -m research new lane_center <slug> "<title>"     # arms are settings of training/configs/base.json (decode, import_weights, every option of the two trainers in stage1 / stage2; from_run through "needs")
python3 -m research start lane_center E<n>               # steps predict + evaluate; a training card names "steps": ["stage1", "stage2", "predict", "evaluate"]
python3 -m research judge lane_center E<n> --write
```
Tests: `python3 -m research tests lane_center` (the list in `problem.json`) and, not yet in that list (`problem.json` is protected),
`python net/tests/test_net_layering.py` from this folder: the layers of `net/` (DESIGN.md R13). Other scripts run from this folder with the
module path of `problem.json` `pythonpath` on `PYTHONPATH`.
