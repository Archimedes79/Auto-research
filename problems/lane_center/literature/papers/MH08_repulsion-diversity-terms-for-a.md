# MH08 - Repulsion / diversity terms for a set of hypotheses (DPP loss; pairwise-distance loss)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L.*

## Papers

Diverse Trajectory Forecasting with Determinantal Point Processes (Yuan, Kitani, ICLR 2020) https://arxiv.org/abs/1907.04967 (ar5iv); A Multi-Loss Strategy for Vehicle Trajectory Prediction: Combining Off-Road, Diversity, and Directional Consistency Losses (2024) https://arxiv.org/abs/2411.19747 (HTML)

## Idea

Add an explicit set-level term that keeps K hypotheses apart. DPP: a kernel from pairwise distances, loss = negative expected cardinality of the DPP. Multi-loss paper: sum of pairwise trajectory distances among drivable modes, applied only during a short fine-tuning.

## Formulation

DPP: similarity S_ij = exp(-k * d_ij^2) from pairwise trajectory distances gives the DPP kernel L; loss = -tr(I - (L + I)^-1) (negative expected cardinality), preferred to the DPP log-likelihood which becomes unstable for similar samples; a "quality" term is defined in latent space (codes inside a sphere containing 90 % of prior samples); k = 1 (synthetic), 1e-2 (mocap), 5 (Human3.6M); the sampling function is trained with the decoder fixed; greedy DPP MAP at inference. Multi-loss: Mode Diversity = sum over drivable pairs (1/T) sum_t ||y_t^i - y_t^j||_2 (sign convention not stated, presumably maximised); plus off-road hinge (signed distance + margin) and a centreline direction-consistency term; weight values not given. - input / output: K trajectories or latent codes.

## Training

added to WTA / minADE; regime `wta`.

## Differentiable?

yes (the kernel / distance is differentiable in the hypotheses).

## Sensors and data

synthetic 2-D vehicle, motion capture, Human3.6M (DPP); Wayformer and AutoBots on nuScenes and Argoverse 2 (multi-loss).

## Reported results

multi-loss, Wayformer on nuScenes: diversity 57 -> 74 while minADE 1.08 -> 1.10, minFDE 2.47 -> 2.55, MR 0.42 -> 0.46; AutoBots / Argoverse 2: minADE 0.92 -> 0.87, diversity 58 -> 62. The paper says adding the diversity loss from the start "tends to destabilize training", so it is used for 10 fine-tuning epochs. DPP paper: DSF beats cVAE, MCL, cGAN on ADE / FDE and on repetitiveness (ASD / FSD).

## Compute

not stated.

## Code

not checked

## Fit to our project

reuse_idea - effort S - a hinge on the child-child distance gated to tips with two GT branches (or to a late stage) is the cheap way to keep fork children apart.

## Needs

pairwise distance matrix of the b children (b is tiny).

## Conflicts

repulsion at normal tips pushes a valid second child off the lane; unstable if active from step 0.

## Tried in our experiments

none
