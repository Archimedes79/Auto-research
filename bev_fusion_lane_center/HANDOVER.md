# HANDOVER - 2026-11-05 20:30

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: one small step per turn, explain terms plainly. **D57: Spurmitte work has priority over everything else, BEV Fusion included.** GPU only in Mirko's windows (nights 22-07, Mon-Fri 08:30-17:30). The BEV system chat knows my plan for tonight (message 20:17).

## What this project is (D50, D55)
A lane head on the frozen GaussianCaR features, end-to-end differentiable (D55: soft path up to the map assignment; hard selection only on the output, named). Shared: `autoresearch/`, `autogt/` (lane GT l3, signature 3576fa1f7e).

## Where we are
**Baseline B (D58)** = E273 head, `runs/e273_p_s0/s1`. VAL 873, mean of 2 seeds: topo F1 0.596, F1@1.0 0.827, junction F1 0.125, lanes found 0.637; reference d 0.741 / 0.878 / 0.273. **E274 closed:** r and p CUT (records committed, `results/E274/README.md`).
**E275 (card `experiments/E275_node-readout-suppression/`, valid):** the close-node suppression (drop a node closer than 1.5 m to a stronger one) is built and tested (`evaluation/qgraph_readout.py`, scorer flags, `verdict275`, 3 test files, 40 mutants, critic pass done; commit c8df1a13). In-sample sweep: B junction F1 0.125 -> 0.26, topo 0.596 -> 0.626. **Prediction 3 missed on the GT alone:** 32.3 % of GT nodes have another GT node within 1.5 m, so the step is not duplicate removal only.

## Running / automatic (started 20:16)
`temp/e275/launch_scoring.sh` (WSL, plain nohup; log `temp/e275/launch.log`) waits for the window and a free GPU (memory < 2.6 GB, no train/eval), then four run.py ids one after the other: E275.n_s0, E275.n_s1 (B, 4 read-outs + radius curve + dump), E275.rn_s0, E275.rn_s1 (r, reported); each with `--repro` against the E274 sweep (tolerance 0.005; stops before any card if missed) and the kill rule (> 20 % of nodes dropped at thr 0.5). About 75 min. Flag `temp/e275/SCORED` at the end. Cards: `runs/e275/<model>.<spec>/`, curves `runs/e275/<model>.curve.json`, dumps `temp/e275/pred_<model>.pkl`.

## Next, by hand, after SCORED
1. `wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh analyse.py e273_p_s0 e273_p_s1 e274_r_s0 e274_r_s1` (CPU; dropped nodes, intersections, pictures in `results/E275/figures/`).
2. Verdict: `qgraph_run_eval.py verdict275 --var e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5 --ctl e273_p_s0 e273_p_s1` (c = the E273 cards, exp e273); n4 / rn / curve are reported, not judged.
3. Ledger notes E275.*, card section 7 (predictions hit / missed), `results/E275/README.md` (needs a `.gitignore` whitelist line as E274), commit by path, refresh tag `lcr/e275-scratch`. Tell the BEV chat when the GPU is free.
4. Then E276: a trained, soft replacement (D55). Report (artifact, version 5, E273 level) still lacks E274 / E275 and the r / p pictures. TEST only with Mirko's recorded OK; nothing pushed.

## Where things are
Report: https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB (private), local `results/E273.report/index.html`; ledger rows E273.*, E274.*; tags `lcr/E274.*`, `lcr/e274-scratch-2`, `lcr/e275-scratch`. Commit by path (`autoresearch/tools/commit_paths.py`).
