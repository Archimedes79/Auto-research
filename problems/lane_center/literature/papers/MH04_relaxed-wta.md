# MH04 - Relaxed WTA (epsilon floor on the non-winners)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L.*

## Papers

Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses (Rupprecht et al., ICCV 2017; authors and venue from memory) https://arxiv.org/abs/1612.00197 (read via ar5iv)

## Idea

Replicate the output layer M times and train with a meta-loss that gives the nearest hypothesis weight 1 - eps and every other hypothesis eps / (M - 1); the small floor keeps starved hypotheses moving until their cell contains data.

## Formulation

loss = sum_m w_m(y) * L(h_m(x), y), w = 1 - eps for the nearest, eps/(M-1) for the others (sum 1); eps = 0.05 in all experiments; additionally full predictions are dropped at random with 1 % probability (as relayed by the page summary). Necessary condition for a minimum: every hypothesis equals the conditional mean of its own cell (centroidal Voronoi tessellation). No score branch; uncertainty read from the hypothesis spread. Dead hypotheses: if all start far from the targets one cell captures every label and only that hypothesis learns. - input / output: shared trunk, M copies of the last layer; any output type (pose, frame, class confidences C * M).

## Training

SGD; regime `wta`; GT only.

## Differentiable?

partial - assignment constant; non-winners get a small gradient.

## Sensors and data

LSP human pose, future-frame prediction (synthetic intersection, NTU RGB-D), VOC / COCO multi-label, VOC2012 segmentation.

## Reported results

4-hypothesis FCN8s on VOC2012 70.3 % mIoU vs 69.1 % for MCL, with 134.9 M vs 539.6 M parameters; pose (strict PCP, oracle): 1 / 2 / 5 / 10 hypotheses 59.7 / 60.0 / 61.2 / 62.8. Cost named: faint "ghost" predictions from centroid bias.

## Compute

only the last layer grows with M.

## Code

not checked

## Fit to our project

reuse_idea - effort S - one scalar; it pulls the second child toward the first (blur), and gives no score.

## Needs

nothing beyond a distance.

## Conflicts

pull toward the cell mean blurs the fork angle when eps is large; aMCL (MH02) argues the bias is permanent.

## Tried in our experiments

none
