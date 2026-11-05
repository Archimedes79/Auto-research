# HANDOVER - 2026-11-05 19:45

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: one small step per turn, explain terms plainly. **D57: Spurmitte work has priority over everything else, BEV Fusion included.** GPU only in Mirko's windows (nights 22-07, Mon-Fri 08:30-17:30). The BEV system chat was told at 18:43 that my chain is finished and the GPU is free; tell it BEFORE I start anything.

## What this project is (D50, D55)
A lane head on the frozen GaussianCaR features, end-to-end differentiable (D55: soft path up to the map assignment; matching in the loss allowed; hard selection only on the output, named). Shared: `autoresearch/`, `autogt/` (lane GT l3, signature 3576fa1f7e).

## Where we are
**Baseline B (D58)** = E273 head, `runs/e273_p_s0/s1`. VAL 873, mean of 2 seeds: topo F1 0.596, F1@1.0 0.827, junction F1 0.125, lanes found 0.637; assignment +0.071 with a 1-2 m map error; reference d 0.741 / 0.878 / 0.273.
**E274 closed (2026-11-05):** arm r (recall weights) CUT (topo -0.039, exact-map guard broken), arm p (r + long-link dual softmax) CUT (junction +0.003 over r). Weights move the operating point only. Records: ledger notes, card section 7, `results/E274/README.md`, tag `lcr/e274-scratch-2`. Gate caution: compare a live phase-end GATE line only with live lines, never with a phase-7 record.
**Finding (in-sample, VAL):** dropping every node within 1.5 m of a stronger one (a hard read-out step) takes B to junction F1 0.260, topo F1 0.626 (thr 0.5); at thr 0.4 topo 0.634, F1@1.0 0.858, lanes found 0.740. R = 2.5 m collapses. The 3x3 cell version was an indexing bug (discarded). Why it works (duplicates?) is not looked at yet.

## Next (card `experiments/E275_node-readout-suppression/`, valid, nothing run)
1. `evaluation/qgraph_readout.py` + scorer flag + tests + `verdict275` wrapper; `ar-critic` code pass.
2. One prediction pass for B s0/s1, r s0/s1 with a dump (about 0.5 GPU h, inside a window, after telling the BEV chat); reproduce the sweep numbers within 0.005.
3. Analysis of the dropped nodes and the radius curve on CPU; 20 pictures. Then the verdict by the written rule.
4. Then E276: a trained, soft replacement of the step (soft suppression inside the forward path, or a repulsion loss). Later: joint-refinement lever (phase 7 lowers junction F1), topo error breakdown, sparsification, temporal fusion. GT ideas only after Mirko says so. TEST only with his recorded OK.

## Not done
The report (artifact below, version 5, E273 level) has no E274 / E275 section yet; no pictures or videos for r and p; HANDOVER over-long before this rewrite.

## Where things are
Report: https://claude.ai/artifact/Vdey4Vj4D6CD942ymy1qwB (private), local `results/E273.report/index.html`; ledger rows E273.*, E274.*; tags `lcr/E274.*`, `lcr/e274-scratch-2`. Nothing pushed (only on Mirko's word, after `git fetch` and `check_git.py`); commit by path (`autoresearch/tools/commit_paths.py`).
