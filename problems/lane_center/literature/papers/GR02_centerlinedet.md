# GR02 - CenterLineDet (RNGDet agent over a predicted BEV raster, nuScenes)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper HTML, config and sampler source); paper and code differ in what they state.. Pipeline stage: S4, S1, S8 (seeds carried between frames).*

## Papers

CenterLineDet: CenterLine Graph Detection for Road Lanes with Vehicle-mounted Sensors by Transformer for HD Map Generation (Xu et al., ICRA 2023) https://arxiv.org/abs/2209.07734

## Idea

Two stages. A BEV network (HDMapNet, or "FusionNet" = HDMapNet + a parameter-free inverse-perspective shortcut with lidar) predicts a centerline heat map and a candidate-seed heat map. Then an RNGDet-type agent walks over the *predicted raster* (not over features) and builds the graph. Heat maps of neighbouring frames are ego-warped and averaged; the dead ends of frame T-1 become seeds of frame T.

## Formulation

Seeds S = peaks of H_I plus endpoints where the agent stopped in frame T-1, minus already explored ones. Agent input = ROI crop around v_t of [averaged H_L, history map M_E]; DETR-like decision transformer returns N-hat vertices with validity; N = 0 stop (endpoint becomes a seed for the next frame), N = 1 continue, N > 1 push all. The paper omits BEV extent, ROI, N-hat and loss equations. Code (`config/get_args.py`, `sampler.py`): BEV +-25 m at 0.25 m (200 x 200), ROI 100 px (= 25 m), expert step 20 px (= 5 m), BC noise 3 px (uniform, added to the executed path, halved on curves, labels stay clean), 10 queries, 6 + 6 layers, hidden 256, validity threshold 0.9, seed filter distance 15 px, endpoint alignment 1 m, lr 1e-3, 100 epochs, batch 1, DETR matcher cost weights (class 1, L1 5, giou 2). Sampler: label = (next GT vertex - v_t) / ROI_half; road mode target = first GT vertex beyond step size (curve if orientation changes > 20 deg within the step); intersection mode uses step / 2 and one target per neighbour; at most 20 steps per agent, 100 per sample; every 3rd frame sampled. A `training_mode` flag allows "DAgger" but the released sampler only contains the BC branch. Inference code (`inference.py`): seeds = centroids of connected components (area > 4 px) of the thresholded (0.5) candidate map, dropped if closer than 15 px to any stored past trajectory; validity = softmax score >= 0.9, in addition a vertex must point forward (dot product with the previous direction > 0), be longer than 1 px, and near-parallel duplicates (dot > 0.99) are removed (the removal branch looks like it never fires); N = 0 ends the direction, N = 1 appends (with snap-to-other-endpoint alignment, which stops the instance), N > 1 spawns one new instance per branch; FIFO queue; caps 40 steps per direction and 200 per frame; trajectories persist in world coordinates across frames and are re-rendered into each frame's history map, so open ends of earlier frames are continued when they lie in the new patch; the multi-frame heat-map fusion is done by a separate script (not in the agent code).

## Input

ROI crop (100 x 100 px) with **two channels fed to the network: predicted centerline segmentation and history map** (code: channels [0, 2] of the stored tensor; the GT-segment channel is only used for labels); `bev_raster`.

## Output

global graph from agent trajectories; up to 10 vertices per step.

## Training

BC on a BFS expert (about 5 h sampling + 1 day training on 4 x RTX 3090); BEV net trained separately; `teacher_forcing` + noise; DETR matching.

## Differentiable?

no - two-stage (BEV net is not trained through the agent), thresholds, queue logic, matching.

## Sensors and data

6 cameras + lidar (FusionNet) or cameras only; nuScenes 698 training sequences, 148 inference sequences (5,981 frames, 2 Hz); BEV extent not in the paper.

## Reported results

nuScenes multi-frame (Table I): CenterLineDet + FusionNet pixel-F1 0.714, topology-F1 0.518 (T-Pre 0.782, T-Rec 0.409) vs FusionNet alone 0.695 / 0.403; no lidar 0.698 / 0.466; no camera 0.605 / 0.394. Topology precision rises a lot, recall barely (0.356 -> 0.409): recall stays bound by the BEV map.

## Compute

0.41 s / frame (2.43 Hz, one RTX 3090), 41 min for 5,981 frames; BEV net alone 13 min.

## Code

https://github.com/TonyXuQAQ/CenterLineDet ; https://tonyxuqaq.github.io/projects/CenterLineDet/

## Fit to our project

baseline + reuse_idea - effort M for a baseline (the agent only needs a heat map, so it can run on our dense head output; blockers: Docker, python-2 evaluation container, CPU-bound sampler); S for the ideas (previous-frame dead ends as seeds, clean labels + noisy executed path, step-halving at junctions).

## Needs

heat map (+ optional seed map), ego poses for warping, GT graph with BFS expert.

## Conflicts

DETR-style bipartite matcher (implied by the matcher cost weights in the config; the paper does not name it); BC only; not end to end; 5 m expert step vs our 2 / 4 / 6 m classes (adapter: step as class).

## Tried in our experiments

E94 - E99 same family (partial); E41 temporal feature fusion gave only +0.008 F1, comparable to their heat-map averaging.
