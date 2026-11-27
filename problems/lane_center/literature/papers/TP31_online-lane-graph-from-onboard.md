# TP31 - Online lane graph from onboard video (early BEV aggregation)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S8.*

## Papers

Online Lane Graph Extraction from Onboard Video (Can et al., ITSC 2023, arXiv 2304.00930) https://arxiv.org/abs/2304.00930

## Idea

Temporal evidence for the STSU-type lane graph is added early: the backbone features of past and future frames are warped into the reference frame's ground-plane BEV grid by a homography (flat-ground assumption, known poses and calibration) and combined by a per-cell maximum, so any number of frames works; the rest of the pipeline is the single-frame model.

## Formulation

per frame, backbone features -> warp to the reference ground-plane BEV grid; binary BEV mask removes warping artefacts; residual block before the max; frames stacked in the batch dimension; per-cell max beat the mean. Training with 3 frames (one past, one future, the reference), test frames 2 s apart; frames sampled within a 4 s window as augmentation. Lane graph as STSU: Bezier centerlines (Q x 3 x 2), incidence matrix A[x,y] = 1 iff the end of x matches the start of y, association feature per query. Baselines: STETR (temporal sinusoidal embeddings in the transformer input) and PostMerge (merge per-frame estimates after inference).

## Input

video frames of one front camera, ego poses; BEV x in [-25, 25] m, forward 1 - 50 m at 25 cm; DeepLabv3+ (Cityscapes).

## Output

Bezier centerlines + connectivity.

## Training

details (optimiser, epochs, loss weights) not reported.

## Differentiable?

partial - Hungarian matching as STSU; the warp and max are differentiable.

## Sensors and data

camera (front); nuScenes and Argoverse.

## Reported results

M-F / Detect / C-F, nuScenes: STSU (1 frame) 56.7 / 59.9 / 55.2; TPLR 58.2 / 60.2 / 55.3; PostMerge (3) 57.3 / 59.9 / 56.0; STETR (3) 57.2 / 60.0 / 61.2; this method 1 frame 58.3 / 60.0 / 61.0, 3 frames 58.8 / 60.3 / 61.3. Argoverse: 3 frames 58.2 / 62.2 / 58.1 vs 1 frame 57.9 / 62.1 / 57.8. Speed 10 FPS (1 frame), 3 FPS (3 frames).

## Compute

see above.

## Code

none ("will be made public")

## Fit to our project

reuse_idea - effort S - independent evidence that explicit multi-frame fusion adds little to a lane-graph metric once a single-frame model is good (+0.5 M-F, +0.3 C-F), like our E41 (+0.008 F1); our cached features are already ego-warpable.

## Needs

relative poses of all frames, flat-ground warp.

## Conflicts

pose error not analysed (odometry assumed exact); flat-ground assumption.

## Tried in our experiments

E39 / E41 (temporal feature fusion).
