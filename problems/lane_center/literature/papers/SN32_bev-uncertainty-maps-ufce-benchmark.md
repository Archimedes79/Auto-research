# SN32 - BEV uncertainty maps: UFCE benchmark, online map uncertainty, direct BEV feature use

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (UFCE, first ~100k characters), A (the two Gu et al. papers)** (A abstract, M method, C code page). Verified: partly. Pipeline stage: E, L.*

## Papers

Predictive Uncertainty Quantification for Bird's Eye View Segmentation: A Benchmark and Novel Loss Function (Yu et al., ICLR 2025; authors from memory) https://arxiv.org/abs/2405.20986 ; Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu, Song, Gilitschenski, Pavone, Ivanovic; CVPR 2024) https://arxiv.org/abs/2403.16439 ; Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention (same authors; ECCV 2024) https://arxiv.org/abs/2407.06683 (both abstract level only)

## Idea

UFCE compares softmax entropy, ensembles, MC dropout and evidential (Dirichlet) heads on LSS, CVT and Simple-BEV and proposes an uncertainty-focal cross-entropy. Gu et al. add per-element uncertainty to online HD-map models and feed it to trajectory prediction (up to 50 % faster training convergence, up to 15 % better prediction on nuScenes), and expose the internal BEV features of the mapping model to the predictor (up to 73 % faster inference, up to 29 % more accurate predictions).

## Formulation

not read in detail.

## Input

BEV logits / features.

## Output

per-cell uncertainty.

## Training

evidential loss; regime matching_free.

## Differentiable?

yes.

## Sensors and data

camera BEV (CARLA, nuScenes, Lyft, nuScenes-C).

## Reported results

nuScenes vehicle, LSS: IoU 0.332 / ECE 0.00887 (entropy, CE) vs 0.343 / 0.00332 (ENN-UFCE); AUPR for misclassification detection stays < 40 %; the paper does not analyse uncertainty versus distance or occlusion.

## Compute

not stated.

## Code

not checked.

## Fit to our project

reuse_idea - effort M - an evidential head on the cached logits would give a learned "unreliable" signal to combine with the geometric mask (SN13); no paper shows that such uncertainty correlates with occlusion or range.

## Needs

logits and labels; extra training.

## Conflicts

none.

## Tried in our experiments

none.
