# MH16 - DiffusionDrive (k-means anchors, nearest-anchor positive, BCE confidences, truncated diffusion)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S5, L.*

## Papers

DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving (Liao et al., CVPR 2025; venue not on the page read) https://arxiv.org/abs/2411.15139 (arXiv HTML)

## Idea

Start the diffusion from noised k-means anchors instead of pure noise (truncated schedule), denoise in 2 steps with a cascade decoder that samples BEV features at the trajectory coordinates, and output a confidence per anchor trajectory; the highest-confidence trajectory is executed.

## Formulation

anchors N = 20 (NAVSIM), 18 (nuScenes); noised anchors tau_k = sqrt(abar_i) a_k + sqrt(1 - abar_i) eps, truncated at 50/1000. Label: the anchor closest to the GT trajectory is positive (y_k = 1), all others negative. L = sum_k [ y_k * L1(tau_hat_k, tau_gt) + lambda * BCE(s_hat_k, y_k) ]: negatives get only the classification term. Decoder: deformable spatial cross-attention to BEV features at the trajectory coordinates, cross-attention to agent / map queries, FFN, timestep modulation; two cascade layers with shared parameters; DDIM update. - input / output: BEV features from the backbone (NAVSIM ResNet-34, nuScenes ResNet-50; the sensor set is not stated on the page read) -> anchor trajectories + confidences.

## Training

100 epochs navtrain, AdamW lr 6e-4, batch 512, 8 x RTX 4090; regime `wta` with fixed anchors + `denoising`.

## Differentiable?

yes (assignment is a precomputed label).

## Sensors and data

NAVSIM (ResNet-34), nuScenes open-loop (ResNet-50).

## Reported results

NAVSIM navtest PDMS 88.1 (Transfuser baseline +4.1); denoising steps 1 / 2 / 3: 87.9 / 88.1 / 88.1; cascade layers 1 / 2 / 4: 87.4 / 88.1 / 88.2; sampled noises 10 / 20 / 40: 84.9 / 88.1 / 88.2; nuScenes open-loop L2 0.57 m, collision 0.08 % (VAD 0.72 m, 0.22 %). The page has no ablation of the number of anchors.

## Compute

about 60 M parameters; 45 FPS (NAVSIM, 4090), planning module 7.6 ms; 8.2 FPS on nuScenes.

## Code

https://github.com/hustvl/DiffusionDrive

## Fit to our project

reuse_idea - effort M - the closest published match of "k-means anchors in metres + nearest-anchor positive + per-anchor BCE confidence + BEV sampling at the hypothesis coordinates" with a nuScenes BEV backbone; a truncated-diffusion refinement is an alternative to E120.

## Needs

anchor set; BEV features sampled at coordinates (we have bilinear sampling).

## Conflicts

the published model trains its BEV backbone end to end (ours is frozen and cached); one positive per sample only (a fork needs two positives).

## Tried in our experiments

none
