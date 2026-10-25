# bev_fusion_lane_center

- Lane centre-line graph from camera + radar BEV features (nuScenes)
- Fully differentiable chain: image -> sparse lane graph -> time -> SD-map route -> planner
- Short graphs, correct graphs, fast (17.5 - 24 ms per frame)
- New chat: read HANDOVER.md
- The problem: PROBLEM.md (goal, chain, metrics, current best, gaps, problem groups, foundation topics); paths for the tools: project.json
- The research system around it: `../autoresearch/` (METHOD.md, agents, templates, prompts, tools); rules for an AI session: CLAUDE.md
- Folder tree with sizes on demand: `tools/maintenance/make_structure.py` (writes `temp/STRUCTURE.md`); structure and document check: `../autoresearch/tools/check_structure.py`

## Folders (by role in the method)
- research - literature/ : OVERVIEW.md (why, nine axes, findings, gaps), CATALOGUE.md (the one reference for 456 papers), papers.csv (the one table to edit), papers/ (404 notes), deep/ + code/ (extractions), register.md (K ids)
- execution state - experiment_execution/ : PLAN.md (experiment tree of the loop), schedule.csv (what next), LEDGER.md (every run), closed_questions.md, runs/ (run records), plans/ (design documents), history/
- experiment summaries - experiments/ : one folder per experiment, E<id>_<slug> (card, detail plan, summary); reports/weekly/
- training - training/ : the product (models, data, losses, train, pipeline, search), trainer.py with checkpoint + resume
- evaluation - evaluation/ : evaluate.py (one entry point, names in models/registry.json), metrics and protocol (sparse_lanes, lane_grid), drivers/, loops/, tests, examples
- tools/ : problem-specific helpers: env launchers, viz, video, catalogue, literature, maintenance (make_structure.py, run_all_tests.sh, CLEANUP_LIST.md); the general tools are in ../autoresearch/tools/
- reports/ : weekly/ (the two weekly reports, two decks, loop 17 draft), build/ (builders), source/ (figures and number digests)
- temp/ : short-lived work
- runs/ raw outputs, models/ checkpoint registry, data/ splits, _to_delete/ staged for the user

## Scripts you need over time
- run a script (retry on CUDA errors): tools/env/_py14.sh <script> (_pyr.sh)
- launch with record, ledger row and code tag: ../autoresearch/tools/run.py (schedule: ../autoresearch/tools/schedule.py)
- train with checkpoint + resume: training/trainer.py
- champion builder, route head, consumer: training/train/p16_train.py, p10_route_train.py, p14_chain_consumer.py
- whole chain: training/pipeline/p16_pipeline.py
- evaluate a named model (all metrics, paired comparison): evaluation/evaluate.py <name> [--compare <name>] (names in models/registry.json); metric code: evaluation/sparse_lanes, dense maps: evaluation/lane_grid
- final, fast, timing evaluation: evaluation/drivers/p16_final_eval.py, p16_fast_val.py, p16_timing.py
- images, videos, tables, reports: tools/viz/render.py, tools/video/make_video.py, tools/catalogue/build_experiment_table.py, reports/build/

## Run
- `wsl bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_fast_check.py --n 20` (fast path = reference graphs, 2 min)
- `wsl bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh sparse_lanes/selftest.py` (metric self-tests)

## Setting of all numbers
- exact poses, simulated ADASIS-like SD map
- scene-disjoint splits: TRAIN 8,414 / VAL 873 / TEST 833 frames
- nuScenes, non-commercial
