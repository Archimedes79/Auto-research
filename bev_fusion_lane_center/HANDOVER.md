# HANDOVER - 2026-11-05 07:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: one small step per turn, explain terms plainly (he did not know what "targets" meant; `results/E273/README.md` has it).

## What this project is (D50, D55)
A separate codebase from the BEV system: a **lane head on the frozen GaussianCaR features**, end-to-end differentiable (D55: soft forward path up to the map assignment; matching in the loss allowed; hard selection only on the output, named). Shared: `autoresearch/`, `autogt/` (lane GT = own version, tag `autogt/lane-l3`, signature 3576fa1f7e; audited clean 2026-11-05, no l4).

## Where we are: E273 run and CUT by hand (2026-11-05, 2 seeds, VAL 873, TEST untouched)
Pixel maps -> one node per 2 m cell -> Hermite links -> soft sections -> assignment to a lane-level map with errors; 24 ms per frame (head). Card section 10 has table, attempts, diagnosis.
- **Holds:** assignment under a 1-2 m map error +0.071 lane accuracy over the geometric control, exact-map guard -0.007. **Broken:** topo F1 0.596 / F1@1.0 0.827 vs d 0.741 / 0.878; junction F1 0.125 vs 0.273; graph 2-3x the vertex budget; section starts 47 % right.
- Why: node recall 0.83 explains the topo gap (oracle graph with 17 % nodes removed: topo F1 0.58, head 0.60; ~90 % recall needed for the guard); the junction gap (0.63 vs 0.13) is the long links (P 0.18 / R 0.49).
- **Next (recommendation, Mirko decides):** node recall, path-level link loss, E271 sparsification on the node chain; then temporal cross-attention, arm Q (CGNet). The assignment module stays.
- GPU free. Models `runs/e273_p_s{0,1}/model.pt`; aborted attempts `runs/_superseded/`.

## Where things are
Report (pictures, videos): artifact https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB (private), local `results/E273.report/index.html`. `results/E273/README.md` (index), `results/E273.p_s{0,1}/`, `runs/e273/<run>/{card.json,results.md,timing.json}`, tags `lcr/E273.p_s0 / p_s1`, `lcr/e273-scratch`, `lcr/e273-scratch-report` (scratch text files; `temp/e273*` stays, ignored). Code: `training/grid/`, `training/train/{qgraph_train,qgates}.py`, `evaluation/{qgraph_eval,qgraph_run_eval,qgraph_timing}.py`, `tools/viz/qgraph_*.py`.

## Not done
Route attribute, temporal fusion, perceived road graph, arm Q, a read-out without the CPU step. All commits local, **nothing pushed** (only on Mirko's word, after `git fetch` and `check_git.py`); commit by path (`autoresearch/tools/commit_paths.py`), other chats have uncommitted files here.
