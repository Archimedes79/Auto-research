# lane_center - Spurmitte

A lane-centre-line graph from camera + radar (GaussianCaR features), end to end differentiable, planner-ready, good and fast. The goal and
its metrics: `PROBLEM.md`; the modules and the rules they obey: `DESIGN.md`; the owner's decisions: `DECISIONS.md`; where things stand:
`HANDOVER.md`; the champion (the two-stage token head on the lane head B') and its numbers: `../../champions/lane_center/`.

| folder | what |
|---|---|
| `net/prep/` | data: the token cache (`lane_tokcache.py`) and its loader (`lane_ptok_data.py`), the frames and the label step of the live token path (`lane_live_data.py`), the frames and the graph ground truth (`frames.py`, `lane_gt_data.py`, `lane_tok_data.py`), targets |
| `net/prep/foundation/` | the GaussianCaR features made again from raw nuScenes (`gcar_features.py`) and a check against the stored ones (`check_features.py`); its `README.md` names what must exist outside the repository |
| `net/model/` | the network as one module (`lane_net.py`: `LaneNet` builds it, loads a checkpoint and runs features -> tokens -> graph; the trainers, the predict step and the evaluation use it), its parts: the lane head B' (`qgraph.py` and its parts), the token head (`jtoken.py`, the pair geometry `jtok.py`), the read-out as a named output step (`lane_readout.py`) |
| `net/loss/` | the token loss (`loss_jtoken.py`) and the losses of B' |
| `training/` | `train/lane_train.py` (the stages from a run's config: its blocks `stage0` (three parts) / `stage1` / `stage2` are the trainers' options, key by key, `tokens` the cache step; test `net/tests/test_lane_train.py`), `train/qgraph_train.py` (stage 0: B' from random weights, its gates `train/qgates.py`; data of its E271 phase `net/prep/grid_data.py`; test `net/tests/test_qgraph_train.py`), `train/jtoken_train.py` (stage 1), `train/jlive_train.py` (stage 2, with B' live in the loop), `trainer.py`, `bflc_paths.py` (where code, data and runs are), `configs/base.json`, the drawing `architecture.*` |
| `evaluation/` | `lane_predict.py` (the step `predict`: the live dump), `lane_dump.py` (the dump's checked reader, on the scored path), `adapter.py` (the evaluation step: decode and score through `Evaluation`), `jtoken_eval.py` (diagnostics: cards, decode sweep, cell rule, timing), `jtiming.py` (the timing protocol), `qgraph_eval.py` + `sparse_lanes/` + `lane_task/` (the scorer and its metrics) |
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

## What the evaluation reports (`evaluation/adapter.py`)
| measure | meaning |
|---|---|
| `graph.topo_F1` (the score), `graph.junction_F1` | connectivity along the driving direction (30 m from seeds every 5 m); forks and merges matched within 3 m |
| `graph.F1@0.5`, `graph.F1@1.0`, `graph.lanes_found`, `graph.false_len_frac` | position of the delivered lines; share of true lanes covered; delivered length off every lane |
| `graph.heading_err_deg` | zig-zag: mean angle between the delivered line and the nearest true lane (delivered length within 1.5 m of one) |
| `graph.dup_frac`, `graph.vert100_dp` | parallel duplicates; vertices per 100 m |
| `cost.ms_median`, `cost.ms_p90` | time per frame of B' + head with its layers, batch 1, RTX 3060 (the foundation features are cached and not part of it); comparable only between runs with `cost.quiet` = 1 |
| `cost.peak_MB`, `cost.params_M` | peak GPU memory of the network (weights + activations of one frame); parameters |

Results of another evaluator or ground truth are not comparable: the judge refuses them and `ledger.csv` marks them (`comparable`). Nothing is
kept compatible for old results; a number that matters is measured again.

## Use (from `autoresearch/`, under WSL)
```
python3 -m research new lane_center <slug> "<title>"     # arms are settings of training/configs/base.json (decode, import_weights, every option of the two trainers in stage1 / stage2; from_run through "needs")
python3 -m research start lane_center E<n>               # steps predict + evaluate; a training card names "steps": ["stage1", "stage2", "predict", "evaluate"]
                                                         # (B' from the repository: "stage0", "tokens" first, with stage0.enabled and tokens.enabled 1)
python3 -m research judge lane_center E<n> --write
```
Tests: `python3 -m research tests lane_center` (the list in `problem.json`), among them `net/tests/test_net_layering.py`: the layers `net/prep` <
`net/model` < `net/loss` < `training/` | `evaluation/` for every import (DESIGN.md R13) and the style of a package, and `net/tests/test_lane_net.py`: the network as one module, `evaluation/tests/test_evaluator_hash.py`: the scored path is the evaluator hash. By
hand on the GPU: `-m lane_center.net.tests.test_lane_net fixture --gpu` (features -> tokens equals the cache bit for bit) and `-m lane_center.evaluation.tests.test_champion_gpu`
(the champion's files give the stored dump and its graphs).

The problem is the Python package `lane_center` (every code folder has an empty `__init__.py`; a module imports another as
`from lane_center.net.model import lane_net as LN`; nothing edits `sys.path`). A module is started as a module, with the module path the research
system gives (`problems/`, `autoresearch/`, the repository), never by its file:
```
python3 -m research py lane_center -m lane_center.net.tests.test_jtoken            # from autoresearch/: the problem's interpreter and module path
PYTHONPATH=..:../../autoresearch:../.. ~/miniconda3/envs/gcar/bin/python -P -m lane_center.net.tests.test_jtoken     # the same from this folder
```
