# bev_fusion_lane_center

Lane centre-line graph from camera + radar BEV features (nuScenes): fully differentiable chain image -> sparse lane graph -> time -> SD-map route -> planner; short graphs, correct graphs, fast (31 ms per frame).
Status 2026-10-24: loops 1-16b done and reported; loop 17 done except E209 and the final report. **Start with [HANDOVER.md](HANDOVER.md)**; rules for an AI session: [CLAUDE.md](CLAUDE.md); names: [CONVENTIONS.md](CONVENTIONS.md); full tree: [STRUCTURE.md](STRUCTURE.md).

## Map
- **[literature/](literature/README.md)** - summary by topic, then detail
  - `papers/` 404 notes (index) - `repos/` 159 repositories + 3 local clones - `register.md` 144 techniques (K ids) - `sources/` original research notes
- **[experiment_execution/](experiment_execution/README.md)** - how experiments are planned, run, logged
  - `PROGRAM.md` current loop - `STATE.md`, `indexes/` state per loop - `plans/` - `LEDGER.md` runs - `logs/` - `templates/` - `history/` (path map, migration log) - `PRACTICE_NOTES.md`
- **[experiments/](experiments/README.md)** - 166 folders `E<id>_<slug>`: README (question, result, links), small results, images, code of finished ones
- **[training/](training/README.md)** - the system: `models/` `data/` (+`prep/`) `losses/` `train/` `pipeline/` `search/`, `trainer.py` (checkpoint + resume)
- **[evaluation/](evaluation/README.md)** - metrics and protocol: `sparse_lanes/` `lane_grid/` `drivers/` `tests/` `examples/` `PROTOCOL.md` `METRIC_RESPONSE.md` `TEST_USAGE_LOG.md`
- **[tools/](tools/README.md)** - `env/` launchers - `experiment/` run.py + ledger.py - `viz/` images - `video/` - `catalogue/` - `maintenance/`
- **[reports/](reports/README.md)** - final PDFs; `build/` builders + figure scripts; `source/` workspace; `per_experiment/`, `background/`, `archive/`
- **[temp/](temp/README.md)** - short-lived: `analysis/loopNN`, `launchers/loopNN`, `archive/` (delete when the loop is filed)
- `runs/` raw outputs ([README](runs/README.md)) - `models/` checkpoint registry - `data/` nuScenes mini + splits ([README](data/README.md)) - `downloads/` - `_to_delete/` staged, not purged ([list](_to_delete/README.md))

## The scripts you need over time
| task | script |
|---|---|
| run anything / retry on CUDA errors | `tools/env/_py14.sh <script>` / `_pyr.sh` |
| launch an experiment with record + ledger | `tools/experiment/run.py`, `ledger.py` |
| train a new model (checkpoint + resume + guard) | `training/trainer.py` (tested: `trainer_selftest.py`) |
| builder (champion D7) train / pipeline | `training/train/p16_train.py`, `training/pipeline/p16_pipeline.py`, `p16_infer.py`, `p16_runtime.py` |
| route head / consumer / dense head | `training/train/p10_route_train.py`, `p14_chain_consumer.py`, `dense_head.py` |
| search harness (successive halving) | `training/search/p16_triage.py` |
| score graphs, compare two systems | `evaluation/sparse_lanes/` (`sparse_metrics.card_any`, `bootstrap.paired`), `evaluation/examples/run_examples.py` |
| score dense maps | `evaluation/lane_grid/grid_eval.py` |
| final / fast / timing evaluation | `evaluation/drivers/p16_final_eval.py`, `p16_fast_val.py`, `p16_fast_check.py`, `p16_timing.py`, `p10_route_eval2.py` |
| result images / videos | `tools/viz/render.py`, `tools/video/make_video.py` |
| catalogue + tables, reports | `tools/catalogue/build_experiment_table.py`, `reports/build/build_loop16_report.py` (template), `finish_report6.ps1` |
| rebuild cached data | `training/data/prep/` |

```bash
wsl bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_fast_check.py --n 20       # champion fast path = reference graphs? (2 min)
wsl bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh sparse_lanes/selftest.py         # metric self-tests (also lane_grid/selftest.py, trainer_selftest.py)
```

Setting of all numbers: exact poses, simulated ADASIS-like SD map, scene-disjoint splits (TRAIN 8,414 / VAL 873 / TEST 833 frames), nuScenes (non-commercial).
