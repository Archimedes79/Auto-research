# HANDOVER - 2026-11-05 12:15

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: one small step per turn, explain terms plainly. **D57 (2026-11-05): Spurmitte work has priority over everything else, BEV Fusion included.** GPU only in Mirko's windows (nights 22-07, Mon-Fri 08:30-17:30).

## What this project is (D50, D55)
A separate codebase from the BEV system: a **lane head on the frozen GaussianCaR features**, end-to-end differentiable (D55: soft forward path up to the map assignment; matching in the loss allowed; hard selection only on the output, named). Shared: `autoresearch/`, `autogt/` (lane GT l3, signature 3576fa1f7e, no l4).

## Where we are
**BASELINE (D58, Mirko 2026-11-05 13:20): the E273 head (`runs/e273_p_s0/s1`, cards `runs/e273/`) is the baseline B** of the lane head; E274 r / p and all later designs are judged against it (E274's control c = B; B is not retrained). E273's "CUT by hand" stays a fact against the suite reference d, it no longer means "do not build on it".
**E273 (now B)** (VAL 873, TEST untouched): assignment +0.071 lane accuracy with a 1-2 m map error (holds), topo F1 0.596 / F1@1.0 0.827 vs d 0.741 / 0.878, junction F1 0.125 vs 0.273; head 24 ms.
**E274 (card `experiments/E274_node-recall-and-long-link-matching/`)**: arm r (existence positives x3 at event cells, x2 inside intersections) trained, 2 seeds, ended 12:00. Node gate (150 frames) passed: node recall 0.67 -> 0.73, precision 0.91 -> 0.89. **Gate-level picture (not the verdict):** topo F1 0.55 vs c 0.60 (-0.045), F1@1.0 0.86 vs 0.83, lanes found 0.77 vs 0.65, junction F1 0.115 vs 0.12, pieces per GT piece 1.9 vs 1.6 = r sits on the c threshold curve (c at node threshold 0.4: topo 0.548, F1@1.0 0.867, lanes 0.783): reweighting moves the operating point, it does not discriminate better. The rule will probably give CUT on the topo guard. Arm p (r + dual-softmax over long links) runs since 12:07, ends about 13:55.

## Running / automatic
`E274.p_s0`, `E274.p_s1` (run.py, logs `runs/e274_p_s<k>_p<phase>/log.txt`). `temp/e274/auto_chain.sh` (detached) starts `chain_after_p.sh` when both are done: scores r_s0, r_s1, p_s0, p_s1 on VAL (exp e274, cards in `runs/e274/`), the three `verdict274` calls (r against c; p against r on junction_F1 band 0.03; p against c), then `sweep_gpu.py` (plain / 3x3 cell NMS / position NMS read-outs for c s0, c s1, r) into `temp/e274/sweep_<run>.json` (hard stop 15:20: the BEV chat starts its GPU part by 15:30, promised "done" before; it creates `temp/lane_tests/GO` only after my message); flags `SCORED`, `CHAIN_DONE`, log `temp/e274/chain.log`. The old CPU NMS sweep was stopped (5 h, no output). After CHAIN_DONE: tell the BEV system chat "GPU free" (it waits; it creates `temp/lane_tests/GO` after that), tell the Image Encoder chats; at 17:00 check nothing of mine would run at 22:00.

## Next, by hand
Ledger notes for E274.* (result, decision), card section 7, pictures/videos for r and p, report update (artifact below) with the E274 section, commit by path, scratch tag `lcr/e274-scratch` refresh. Design after the numbers: if NMS recovers topo at higher recall, name it as the read-out step and look for a soft replacement; else the lever is a cleaner existence map (more evidence, e.g. the map prior inside intersections), not weights. Later: sparsification arm, temporal cross-attention, route attribute, CGNet reference; GT ideas (events with painted evidence, occlusion by boxes) only after Mirko says so.

## Where things are
Report: artifact https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB (private, version 5, E273 level), local `results/E273.report/index.html`; `results/E273/README.md`; ledger rows E273.*, E274.*; tags `lcr/E273.*`, `lcr/e273-scratch*`, `lcr/e274-scratch`. Nothing pushed (only on Mirko's word, after `git fetch` and `check_git.py`); commit by path (`autoresearch/tools/commit_paths.py`).
