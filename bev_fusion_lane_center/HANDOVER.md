# HANDOVER - 2026-11-05 06:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko is often interrupted and reads little: one small step per turn, say where we are, explain terms (he did not know what "targets" meant: `results/E273/README.md` has the plain-words version).

## What this project is (D50, D55)
A separate codebase from the BEV system: a **lane head on the frozen GaussianCaR features**, end-to-end differentiable (D55: soft forward path up to the map assignment; matching inside the loss is allowed; hard selection only on the output, named). Shared: `autoresearch/`, `autogt/` (the lane GT is its own version, tag `autogt/lane-l3`, signature 3576fa1f7e; the front GT c643a7539d is untouched).

## Where we are: E273 is run and CUT by hand (2026-11-05, both seeds, VAL 873, TEST never touched)
Pixel maps (0.5 / 1 / 2 m) -> one node per 2 m cell -> Hermite links -> soft sections -> assignment to a lane-level map with simulated errors. Built, trained (2 seeds, 24 ms per frame for the head) and scored overnight; card section 10 has the table, the attempts, the diagnosis.
- **Holds:** assignment under a 1-2 m map error +0.071 lane accuracy over the geometric control (interval excludes 0 and the band), exact-map guard -0.007. **Broken:** graph guards, topo F1 0.596 / F1@1.0 0.827 against d 0.741 / 0.878 (junction F1 0.125 against 0.273; 2-3x the vertex budget; section starts only 0.47 right).
- Why: node recall 0.83 caps the topology; long links through intersections have P 0.18 / R 0.49 (false and missed connectors around every junction).
- **Next (recommendation, Mirko decides):** (1) node recall and event nodes first, (2) a path-level link loss, (3) the E271 sparsification on the node chain, then the follow-ups (temporal cross-attention to earlier nodes, arm Q = CGNet as a reference). The assignment module stays.
- GPU is free; nothing of this project runs. Seed folders `runs/e273_p_s0`, `runs/e273_p_s1` (model.pt), aborted attempts in `runs/_superseded/`.

## Where things are
`results/E273/README.md` (index: verdict, numbers, targets in plain words, GT defaults, versions, rebuild), `results/E273.p_s{0,1}/` (README, package.json; figures / videos not in git: start with `videos/scene-0221_*.mp4`), `runs/e273/<run>/{card.json,results.md,timing.json}`, ledger rows `E273.p_s0 / p_s1`, tags `lcr/E273.p_s0 / p_s1` and the scratch tag of this loop. Code: `training/grid/`, `training/data/lane_gt_data.py`, `training/train/{qgraph_train,qgates}.py`, `evaluation/{qgraph_eval,qgraph_run_eval,qgraph_timing}.py`, `tools/viz/{qgraph_view,qgraph_results}.py`.

## Not done / named gaps
Route attribute, temporal fusion, perceived road graph, arm Q, a read-out that is not a CPU step; the packager's `eval/metrics.json` was written by a scratch script. Everything is committed locally, **nothing pushed** (push only on Mirko's word, after `git fetch` and `check_git.py`); other chats have uncommitted files in this tree: commit by path (`autoresearch/tools/commit_paths.py`).
