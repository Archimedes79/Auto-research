# HANDOVER - 2026-11-05 22:10

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: one small step per turn, plain terms, no letter clashes (B = baseline only, tiers are 1 / 2 / 3). **D57: Spurmitte first.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). The BEV system chat knows tonight's plan.

## Project (D50, D55)
Lane head on frozen GaussianCaR features, end-to-end differentiable (soft up to the map assignment; matching in the loss allowed; hard selection only on the output, named). Shared: `autoresearch/`, `autogt/` (lane GT l3, signature 3576fa1f7e).

## Where we are
**Baseline B (D58)** = E273 head, `runs/e273_p_s0/s1`. VAL 873, mean of 2 seeds: topo F1 0.596, F1@1.0 0.827, junction F1 0.125, lanes found 0.637; reference d 0.741 / 0.878 / 0.273. E274 closed (r, p CUT).
**E275** (`experiments/E275_node-readout-suppression/`): close-node suppression (drop a node < 1.5 m from a stronger one) built, tested, critic pass (c8df1a13). In-sample: junction F1 0.125 -> 0.26, topo 0.596 -> 0.626. Prediction 3 missed on the GT alone: 32.3 % of GT nodes have a GT neighbour < 1.5 m.
**D59 (Mirko, 22:0x):** relevance tiers in the GT (1 = ego lane along the driven path, 2 = lanes left / right of it, 3 = rest, plus a visibility flag) and in the loss (weights by tier, wrong costs more than missing, unseen junction branches not demanded). Open: weight values, an "unknown" output flag, PROBLEM.md metrics. **Next small step:** tier labels as a tested CPU module (path from the following ego poses, left / right from the road graph, visibility from autogt coverage), pictures on real VAL frames, critic; then a loss switch in `CFG`, then the card (E276 line).

## Running (started 22:00:26)
`temp/e275/launch_scoring.sh` (log `temp/e275/launch.log`): E275.n_s0 runs, then n_s1, rn_s0, rn_s1 (about 75 min; flag `temp/e275/SCORED`). `--repro` (0.005 against the E274 sweep) and the kill rule (> 20 % nodes dropped at thr 0.5) stop it before any card. Cards `runs/e275/<model>.<spec>/`, dumps `temp/e275/pred_<model>.pkl`. Cron check 23:41.

## After SCORED
1. `wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh analyse.py e273_p_s0 e273_p_s1 e274_r_s0 e274_r_s1` (CPU).
2. `qgraph_run_eval.py verdict275 --var e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5 --ctl e273_p_s0 e273_p_s1`.
3. Ledger notes E275.*, card section 7, `results/E275/README.md` (+ `.gitignore` line), commit by path, refresh tag `lcr/e275-scratch`, tell the BEV chat the GPU is free.
Report (artifact v5) lacks E274 / E275. TEST only with his OK. Nothing pushed.

## Where
Report https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB (private), `results/E273.report/index.html`. Ledger E273.*, E274.*; tags `lcr/E274.*`, `lcr/e274-scratch-2`, `lcr/e275-scratch`. Commit by path (`autoresearch/tools/commit_paths.py`).
