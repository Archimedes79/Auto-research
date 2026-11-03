# HANDOVER - 2026-11-03 10:30

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko is often interrupted and reads little: one small step per turn, say where we are.

## What this project is (D50, Mirko 2026-11-03)
A separate codebase from the BEV system: a **lane-centre head on the frozen GaussianCaR features** (E2E differentiable, no rule step on the training path). The BEV system is not its trunk; nothing here depends on `bev_fusion/` and nothing there on this folder. Shared: `autoresearch/` (tools, trainsuite, evalsuite), `autogt/`.

## Who does what
- **Lead chat "BEV Fusion Spurmitte"** (this file): shared code of the folder (`training/trainer.py`, tools, docs), reviews, decisions. **"BEV Fusion Spurmitte Experiments"**: the E271 head (`training/grid/`, `training/train/grid_train.py`, `evaluation/vertex_eval.py`, `lane_task/measures.py`): do not edit its files. GT chat: `autogt/`.

## Where we are
- Prototype of record (the chain with rule steps): `recall_sp0_dp` (VAL lanes found 0.781, Q 0.742; TEST E268.p 0.740 / 0.727). D4 waits for route success and idle timing.
- E271 grid head (1 m centre-lane head + differentiable sparsification, 39 frozen GaussianCaR channels): arm `a_s0` done (VAL cov5 0.721, prec1 0.871, 10.3 vertices per 100 m of GT lane; reference r 0.591 / 0.701 / 11.2, null 0.14 / 0.02); with the read-out threshold at its floor the head reaches only cov5 0.744 at 10.9: coverage is capped, not the budget. `a_s1` runs (GPU slot 0). Verdict by `decide.py` after both seeds (card `experiments/E271_*`).
- Lane tooling (own, committed 2026-11-03): trainsuite contract in `training/trainer.py` + `training/probe.py`, `training/startable.py` + `training/configs/` + `train/run_config.py` + project.json `trainsuite` block, fast gates `wsl bash tools/maintenance/run_gates.sh` (full suite `run_all_tests.sh`); `training/legacy` and dead scripts removed (tag `lcr/pre-restructure-2026-11-03`).

## Next
1. E271 verdict (both seeds), then why coverage stops at 0.74 (by range / situation: `lane_suite.py dense:<run>`), then the hold (stage 3) and the assignment (stage 4) as their own cards.
2. A critic pass (mode code) on the grid head code before more arms.
3. Open for Mirko: D4, D5. Commit by path (`autoresearch/tools/commit_paths.py`); other chats have uncommitted files in this tree.

## Pointers
PROBLEM.md; experiment_execution/{DECISIONS,PLAN,LEDGER}.md; training/README.md, training/grid/README.md; git: tag `lcr/<id>` per run, pushes only on request.
