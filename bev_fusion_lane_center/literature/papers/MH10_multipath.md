# MH10 - MultiPath (fixed anchors: classify the anchor, regress the offset)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, S4.*

## Papers

MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction (Chai, Sapp, Bansal, Anguelov, CoRL 2019) https://arxiv.org/abs/1910.05449 (arXiv HTML)

## Idea

Fix K anchor trajectories before training; the model outputs a softmax over anchors and, per anchor, a Gaussian offset. Each GT is assigned to the single nearest anchor, so the assignment is a fixed geometric rule and the mixture cannot collapse.

## Formulation

k_hat = argmin_k ||GT - a^k||_2 (state-sequence space, canonical agent frame). Loss = -[ log pi(a^{k_hat} | x) + sum_t log N(s_t | a_t^{k_hat} + mu_t, Sigma_t) ] (Eq. 3; regression only for the assigned anchor; with Sigma frozen an L2 loss). Mixture p(s | x) = sum_k pi(a^k | x) prod_t N(s_t | a_t^k + mu_t^k, Sigma_t^k), weights fixed over time. Anchors: k-means on training trajectories (redundant on some data -> uniform sampling of trajectory space). K = 16 (mu, Sigma) or 64 (mu) on the driving data; K = 3 toy; Stanford Drone 64 straight + 1 stationary. Inference: MAP trajectory per anchor weighted by pi; keep the top kappa (no NMS / EM). - input / output: rasterised scene + state history -> K anchor probabilities + offsets.

## Training

500 k steps, batch 32; regime `wta` with a fixed (anchor) assignment.

## Differentiable?

yes for all heads; the anchor label is precomputed.

## Sensors and data

an internal driving dataset (not public), Stanford Drone; BEV raster.

## Reported results

Table 1 on the driving data: log-likelihood 4.37 (MultiPath mu, Sigma) vs 4.26 (min-of-K) vs 3.64 (regression); minADE5 0.58 (MultiPath mu) vs 0.87 (min-of-K mu). Toy: min-of-K collapsed to 1 or 2 active modes across trials.

## Compute

one forward pass; K up to 64.

## Code

not checked

## Fit to our project

reuse_idea - effort S - fixed bearing / step-class anchors in the tip frame give every child slot a fixed role; the E94 head is already an anchor-style classifier.

## Needs

an anchor set in the tip frame (bearings, step classes); a cross-entropy target "nearest anchor".

## Conflicts

needs K >> b (16 - 64 anchors) and then a top-b with suppression; anchors fix the mode count; [own inference] a hard nearest-anchor label flips at sector borders, and a straight lane lies exactly on the border between left and right sectors.

## Tried in our experiments

E94 (partial: bearing-bin + step-class classification, teacher forced, one child + fork event); verdict F1@1 0.851 vs 0.867 for rules
