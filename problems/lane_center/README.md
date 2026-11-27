# lane_center - Spurmitte

A lane-centre-line graph from camera + radar (GaussianCaR features), end to end differentiable, planner-ready, good and fast. The goal and
its metrics: `PROBLEM.md`; the modules and the rules they obey: `DESIGN.md`; the owner's decisions: `DECISIONS.md`; where things stand:
`HANDOVER.md`. Champion (D120): E372 S4, the two-stage token head on the frozen lane head B' (`champions/lane_center/`).

| folder | what |
|---|---|
| `net/prep/` | data: the token cache (`lane_tokcache.py`), the token loaders (`lane_ptok_data.py`, `lane_tok_data.py`), the GT geometry of the E395 position head (`lane_gtgeom_data.py`, default off), features and graph ground truth (`frames.py`, `lane_gt_data.py`), targets |
| `net/model/` | the network: the token head (`jtoken.py`; the pair geometry `jtok.py`), B' (`qgraph.py` and its parts), the soft read-out pieces |
| `net/loss/` | the token loss (`loss_jtoken.py`) and the losses of B' |
| `training/` | `train/jtoken_train.py` (stage 1), `train/jlive_train.py` (stage 2), `train/lane_live_tok.py` (B' live in the loop), `train/lane_train.py` (the two stages from a run's config), `trainer.py`, `bflc_paths.py` (where code, data and runs are), `configs/base.json`, the drawing `architecture.*` |
| `evaluation/` | `lane_predict.py` (the step `predict`: the live dump), `adapter.py` (the evaluation step: decode and score through `autoresearch/Evaluation`), `jtoken_eval.py` (dumps, decodes, diagnostics), `jtiming.py` (the timing protocol of its `timing`), `qgraph_eval.py` + `sparse_lanes/` + `lane_task/` (the scorer and its metrics) |
| `tools/viz/` | pictures of graphs over BEV and camera (`qual_frames.py`, `qgraph_view.py`) |
| `experiments/` | from E396 on, one folder per experiment (`research new lane_center <slug> "<title>"`) |
| `data/splits.json` | the splits by row: VAL 873 frames in 22 scenes, TEST 833 frames in 21 scenes (holdout) |

## Data and runs (not in git; where the machine file says)
- `<data>/lane_center/b123/`: the GaussianCaR features (`feats/{train,val}_{dec32,logit7}.npy`, tokens), the graph ground truth the
  headline numbers are scored against (`graph_gt_{train,val}.pkl`), poses (the scene of a frame); a re-extraction is not bit-identical, so
  they are kept.
- `<data>/groundtruth/lane/9ad1d4ce30/`: the lane ground truth l6 of GtGenerator; `<data>/nuscenes/`: raw nuScenes (cameras for pictures).
- `<runs>/lane_center/`: the runs of the experiments, and the imported networks of the champion's chain under their old names
  (`e372_s4_s*`, `e373_cb_s*`, `e292_a_s*` = B', `e273_p_s*`, `e271_a_s*`), the token cache `e365_cache/`, the label tables `e357_cache/`,
  `e365_planb/`, `refiner_stats_dec32_logit7.npz`.

## Use
```
python -m research new lane_center <slug> "<title>"   # a card: arms are settings of training/configs/base.json (decode, stage1, stage2, import_weights)
python -m research start lane_center E<n>             # steps predict + evaluate by default; a training card names "steps": ["stage1", "stage2", "predict", "evaluate"]
python -m research judge lane_center E<n> --write
```
Tests: `research tests lane_center` (the list in `problem.json`), and `python net/tests/test_net_layering.py` (R13; not in that list until the
owner adds it to `problem.json`, a protected file). By hand: from this folder, the module path of `problem.json` "pythonpath" on `PYTHONPATH`.
The research notes of the older literature work are in the tag `v1`.

What is not here: the code of the earlier chain (loops 8 - 18, E324 s, the junction and route heads, E357's pair head, E368's past-frame data,
the old lane suite, the old reports and their builders) and
of the stages before B' (its weights are kept, its training code of E271 / E273 is lost since 2026-11-10): all of it is in the tag `v1`.
