# HANDOVER - 2026-11-07 18:50

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: one small step per turn, plain terms (B = baseline only; tiers 1 / 2 / 3). **D57: Spurmitte first.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30).

## Project (D50, D55)
Lane head on frozen GaussianCaR features, end-to-end differentiable (matching in the loss allowed; hard selection only on the output, named). Shared: `autoresearch/`, `autogt/` (lane GT l3).

## Where we are
**Baseline B (D58)** = E273 head, `runs/e273_p_s0/s1`. VAL 873: topo F1 0.596, junction F1 0.125 (P 0.07, R 0.65), lanes found 0.637; d 0.741 / 0.273. Card seed 0: `vert100_dp` 28.9 vs budget 14.3; end events P/R 0.36 / 0.35 (3 m); assignment lane_acc 0.98 per node (blind to duplicates). E274 closed (r, p CUT).
**E275** (close-node suppression, `experiments/E275_node-readout-suppression/`): built, tested (c8df1a13); in-sample junction F1 0.125 -> 0.26. Scoring attempt 1 died 11-05 22:11 on a path bug (fixed a5e52acc). **Re-armed 11-07 ~18:30:** `temp/e275/launch_scoring.sh 150` (WSL, log `temp/e275/launch.log`) waits for 22:00, then n_s0, n_s1, rn_s0, rn_s1 (~75 min, flag `temp/e275/SCORED`). BEV chat told.
**D59:** tiers in the GT (1 ego lane along the driven path, 2 left / right lanes, 3 rest, + visibility) and in the loss. **Next small step:** tier-label module (CPU, test, pictures on VAL).
**Mirko's notes on the videos (11-07):** intersection label not always good; assignment looked bad (1:1?); no temporal consistency at junctions; endpoints?; node density too high; are the metrics good?; own lane much more important, non-causal GT ok, no demand without visibility; show uncertainty of links. Ideas: GT nodes by Douglas-Peucker (GT has a node every 2 m now), tier + visibility weights, flicker and endpoint measures, calibrated link p as opacity, risk-coverage curve.

## After SCORED
1. `wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh analyse.py e273_p_s0 e273_p_s1 e274_r_s0 e274_r_s1`.
2. `qgraph_run_eval.py verdict275 --var e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5 --ctl e273_p_s0 e273_p_s1`.
3. Ledger notes E275.*, card section 7, `results/E275/README.md` (+ `.gitignore`), commit by path (`autoresearch/tools/commit_paths.py`), refresh tag `lcr/e275-scratch`, tell the BEV chat the GPU is free. TEST only with his OK; nothing pushed. Report (artifact v5, https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB) lacks E274 / E275.
