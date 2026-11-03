# HANDOVER - 2026-11-03 10:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko is often interrupted and reads little: one small step per turn, say where we are.

## Who does what
- **Lead chat "BEV Fusion Spurmitte"** (this file's owner): restructure of this folder in the BEV system's way (plan: `experiment_execution/plans/restructure_like_bev_2026-11-03.md`), shared code of the folder (`training/trainer.py`, tools, docs), decisions. **"BEV Fusion Spurmitte Experiments"**: the E271 head (`training/grid/`, `training/train/grid_train.py`, `evaluation/vertex_eval.py`, `lane_task/measures.py`): do not edit its files. **BEV system chat** (lead of `bev_fusion/`, plan `../bev_fusion/plan/plan_2026-11-03.md`, D48): restructures `autoresearch/` (a `trainkit`, one evaluation front door, manifests, splits) and wants the grid head as the 1 m centre-lane head of its bundled experiment t20; lane's `trainer.py` must stay a subset of the trainkit contract. GT chat owns `autogt/`, tools chat `autoresearch/tools`.

## Where we are
- Prototype of record: `recall_sp0_dp` (VAL lanes found 0.781, Q 0.742; TEST E268.p: 0.740 / 0.727). D4 (promote it) waits for route success and idle timing. D47: D26 clause "dense head first, rule graph at inference" withdrawn; lane head is E2E only; regions are data; sparse lanes become a shared GT vector layer (autogt, GT chat).
- E271 (grid head 1 m + differentiable sparsification vs the prototype's vertices): code, oracle G0, arm `e271_a_s0` training (GPU slot 0), `a_s1` next; card `experiments/E271_*`. Verdict by `decide.py` after both seeds.
- Restructure done today (commits 38d7c51, 7cd81b8, 54ebc03): trainsuite contract in `training/trainer.py` + `training/probe.py` (copied from bev_fusion), RNG-on-GPU resume bug fixed, `training/legacy` and dead scripts removed (tag `lcr/pre-restructure-2026-11-03` restores them), `training/startable.py` + `training/configs/` + `train/run_config.py` + project.json `trainsuite` block (`python -m trainsuite start training/configs/e271_grid.json --id <id> --preset seed1 ...`), fast gates `wsl bash tools/maintenance/run_gates.sh` (full suite `run_all_tests.sh`, baseline 2026-11-03: all pass except the autoresearch selftest, which sees other chats' live runs).

## Next
1. Not yet done of the plan: board / module probe / idle-GPU timing tools in `evaluation/` (when two heads are trained), `records/` (champion.json, board), freeze of the frame after the E271 baseline, `experiment.json` / manifest with the BEV chat's plan (wait for their answers), the autogt lane_graph layer (GT chat) and a region argument for the graph metric (D47).
2. Open for Mirko: D4, D5; TEST-scene overlap rule (the lane VAL / TEST scenes are inside BEV's 82 val scenes) before any lane head runs on a BEV trunk (D47 (d)).
3. Commit by path (`autoresearch/tools/commit_paths.py`), never `git add -A`; other chats have uncommitted files in this tree.

## Pointers
PROBLEM.md; METHOD.md; experiment_execution/{DECISIONS,PLAN,LEDGER}.md; training/README.md; git: tag `lcr/<id>` per run, pushes only on request.
