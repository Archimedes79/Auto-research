# HANDOVER - 2026-11-04 23:30

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko is often interrupted and reads little: one small step per turn, say where we are.

## What this project is (D50, D55)
A separate codebase from the BEV system: a **lane head on the frozen GaussianCaR features**, end-to-end differentiable (D55, 2026-11-04: soft forward path up to the map assignment; matching inside the loss is allowed; hard selection only on the output, named). Nothing here depends on `bev_fusion/`. Shared: `autoresearch/`, `autogt/` (the lane GT is its own version `autogt.lane_gt`, tag `autogt/lane-l3`, signature 3576fa1f7e, built by the GT chat; the front GT c643a7539d is untouched).

## Where we are: E273 (card `experiments/E273_*/experiment.md`, revision 6 = the full plan)
Pixel maps (0.5 / 1 / 2 m) -> one node per 2 m cell (type flags, pulled onto the line) -> Hermite links (short: chord / 3; long through intersections: learned handle scales) -> sections (soft absorbing walk, ids only at splits / merges) -> assignment to a lane-level map with simulated errors (cross-attention + E272's geometric term + dustbin + soft registration). Code: `training/grid/{qtargets,qformat,map_sim,qraster,pyramid,qnodes,qlinks,sections,qassign,qgraph,qpredict}.py`, `training/data/lane_gt_data.py`, `training/train/{qgraph_train,qgates}.py`, `evaluation/{qgraph_eval,qgraph_run_eval}.py` (the scorer reproduces the lane suite numbers of recall_sp0_dp exactly), `tools/viz/{qgraph_view,qgraph_results}.py`. Oracle on real GT frames: lane points 99.9 %, connectors 98.7 % within 1 m.
- Run `E273.p_s0` (all phases 2-7, a gate after each; `temp/e273/start_p.sh 0`): started 2026-11-04 23:09; phase logs `runs/e273_p_s0_p<k>/log.txt`, gates in the last lines. Seed 1 after it (`start_p.sh 1`). Then `qgraph_run_eval.py score --run e273_p_s<k> --seed <k>`, `verdict --runs e273_p_s0 e273_p_s1` (by hand, the card's rule), pictures `qgraph_results.py --run ...`, result folder `results/E273/` (README + package.json in git, figures / videos not).
- Reference d = recall_sp0_dp: Q 0.742, F1@1.0 0.878, topo F1 0.740, junction F1 0.273. Control g = E272 chain + geometric assigner on the same map.

## Not done / named gaps
Route attribute, temporal fusion (follow-up: cross-attention to earlier nodes), perceived road graph, ms per frame per stage, TEST (never touched). Vertex budget: node ids 33 per 100 m, Douglas-Peucker 11.7 (budget 14.3 on the DP basis).

## Pointers
PROBLEM.md (D55 wording); experiment_execution/{DECISIONS,PLAN,LEDGER}.md; training/grid/README.md; git: tag `lcr/<id>` per run, pushes only on request. Commit by path (`autoresearch/tools/commit_paths.py`); other chats have uncommitted files in this tree.
