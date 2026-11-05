# HANDOVER - 2026-11-05 08:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: one small step per turn, explain terms plainly. **D57 (2026-11-05): Spurmitte work has priority over everything else, BEV Fusion included.**

## What this project is (D50, D55)
A separate codebase from the BEV system: a **lane head on the frozen GaussianCaR features**, end-to-end differentiable (D55: soft forward path up to the map assignment; matching in the loss allowed; hard selection only on the output, named). Shared: `autoresearch/`, `autogt/` (lane GT l3, signature 3576fa1f7e, audited clean 2026-11-05, no l4).

## Where we are
**E273 CUT by hand** (2 seeds, VAL 873, TEST untouched): assignment +0.071 lane accuracy with a 1-2 m map error (holds), topo F1 0.596 / F1@1.0 0.827 vs d 0.741 / 0.878 (guards broken), junction F1 0.125 vs 0.273; head 24 ms.
**Diagnostics 2026-11-05** (`temp/e274/`, `temp/e273/report/`): node recall explains the topo gap (oracle with 17 % nodes removed: 0.58, head 0.60); true nodes are lost as gaps (median 3.8 m to the nearest delivered node), 55 % of the misses inside intersections (recall 0.73 vs 0.94 outside), events weak (pixel event F1 0.37-0.57); long links P 0.18 / R 0.49 are the junction problem.
**E274 (card `experiments/E274_node-recall-and-long-link-matching/`)**: arm r = recall weights (event loss x3, existence positives x3 at events, x2 in intersections), arm p = r + dual-softmax loss over the long links; control c = E273 checkpoints. Code switches in `training/grid/qgraph.py` (`CFG`), flags in `qgraph_train.py`, `qgraph_run_eval.py verdict274`. Launch: `temp/e274/launch_window.sh` (waits for the window Mon-Fri 08:30-17:30 / nights, the flag `temp/e274/READY` and a free GPU; r_s0 + r_s1, then p after them).

## Next
Critic pass result -> READY flag -> r training (about 2.4 h per seed) -> `qgraph_run_eval.py score --run e274_r_s<k> --seed <k> --exp e274` -> `verdict274` -> p (about 1.3 h) -> pictures, videos, report. Later: sparsification arm, temporal cross-attention, route attribute, CGNet reference; GT ideas (events with painted evidence, occlusion by boxes) only after Mirko says so.

## Where things are
Report (strategy, architecture + literature, GT, results, videos): artifact https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB (private), local `results/E273.report/index.html`; `results/E273/README.md`; ledger rows E273.p_s0/p_s1, E274.*; tags `lcr/E273.*`, `lcr/e273-scratch`, `lcr/e273-scratch-report`. Nothing pushed (only on Mirko's word, after `git fetch` and `check_git.py`); commit by path (`autoresearch/tools/commit_paths.py`).
