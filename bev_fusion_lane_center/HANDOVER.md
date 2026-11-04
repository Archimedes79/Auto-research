# HANDOVER - 2026-11-04 14:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko is often interrupted and reads little: one small step per turn, say where we are.

## What this project is (D50, Mirko 2026-11-03; restated 2026-11-04: "GaussianCaR is right now the development foundation")
A lane-centre head on **GaussianCaR** (frozen camera + radar BEV features, 39 channels `dec32` + `logit7`, cached in `~/exp_gcar_b123/feats`), E2E differentiable, no rule step on the training path. GaussianCaR is the foundation for all development and every number here. The `bev_fusion/` system is a separate codebase and not this head's trunk (it cannot yet extract lanes well: lane divider IoU@1 0.55); nothing here depends on it. Shared: `autoresearch/` (tools, trainsuite, evalsuite), `autogt/`.

## Who does what
- **Lead chat "BEV Fusion Spurmitte"**: shared code of the folder (`training/trainer.py`, tools, docs), reviews, decisions. **"BEV Fusion Spurmitte Experiments"**: the head (`training/grid/`, `training/train/grid_train.py`, `evaluation/vertex_eval.py`, `lane_task/measures.py`): do not edit its files. GT chat: `autogt/`.

## Where we are
- Prototype of record (chain with rule steps): `recall_sp0_dp` (VAL lanes found 0.781, Q 0.742; TEST E268.p 0.740 / 0.727). D4 (route success, idle timing) waits for Mirko.
- E271 1 m head + differentiable sparsification: KEEP by hand (VAL 873 frames, cov5 0.721 / 0.728, prec1 0.871, ~10.3 vertices per 100 m; reference r 0.591). E272 chain (head, sparsify, hold, assignment) as one function: TIE by hand on both seeds (cov5 +0.006), accepted as THE end-to-end function. Cached GaussianCaR features only, VAL only; the head has never seen TEST.
- Gaps named: the hold switches itself off on clean frames (value untested: needs degraded frames); no delivered lane graph (lane ends, edges); coverage stops near 0.74; no live check through the real GaussianCaR forward pass; budget-matched strong baseline (GT resampled every 10 m) missing; ms not measured.
- Lane tooling (own): trainsuite contract in `training/trainer.py` + `probe.py`, `startable.py` + `configs/` + `train/run_config.py`, fast gates `wsl bash tools/maintenance/run_gates.sh`.

## Next
1. Live check on ~100 VAL frames: raw camera + radar -> GaussianCaR checkpoint -> head, against the cached numbers (recipes `training/data/prep/`).
2. Degraded-frame test of the hold; delivered lane graph card; coverage by range / situation (`lane_suite.py dense:<run>`).
3. A critic pass (mode code) on the grid head code. TEST only with Mirko's recorded OK. Open for Mirko: D4, D5, headline_metrics override for vertex-level cards (protected project.json).

## Pointers
PROBLEM.md; experiment_execution/{DECISIONS,PLAN,LEDGER}.md; experiments/E271_*, E272_*; training/README.md, training/grid/README.md. Commit by path (`autoresearch/tools/commit_paths.py`); other chats have uncommitted files in this tree. Pushes only on request.
