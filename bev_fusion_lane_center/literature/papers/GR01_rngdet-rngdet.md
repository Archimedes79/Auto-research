# GR01 - RNGDet / RNGDet++ (DETR-style set-of-next-vertices agent, behaviour cloning)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper HTML, README); RNGDet++ ablation row mapping not recoverable.. Pipeline stage: S4 (per-step hypotheses), S1 (seeds), T.*

## Papers

RNGDet: Road Network Graph Detection by Transformer in Aerial Images (Xu et al., IEEE T-GRS 2022) https://arxiv.org/abs/2202.07824 ; RNGDet++: Road Network Graph Detection by Transformer with Instance Segmentation and Multi-scale Features Enhancement (Xu et al., RA-L 2023) https://arxiv.org/abs/2209.10150

## Idea

An agent crawls the road graph from seed vertices. At each vertex it crops a window of image features, two segmentation maps (road, intersection / endpoint) and a raster of the graph built so far; a DETR transformer with 10 learned queries proposes up to 10 next vertices, each with a validity score. The count M of valid outputs decides: 0 = dead end (take next seed), 1 = move on, >1 = junction (all outputs go to a FIFO buffer). Trained only by behaviour cloning on a BFS expert whose executed path is perturbed by noise.

## Formulation

L = L_seg + 5 L_coord + 1 L_valid (RNGDet). L_coord = L1 over bipartite-matched (prediction, GT) pairs; L_valid = BCE (matched = 1). RNGDet++ adds L_ins (BCE of a per-query mask of the road segment between v_t and the proposed vertex; weights garbled in the fetched text) and 4-level FPN features. Label rule on the GT graph after removing degree-2 vertices: road-segment mode, one target = GT point tau = 40 px ahead, or the nearest sharp turn, or a seed if one lies within tau; intersection mode (after a segment ended), one target per incident segment at tau' = 20 px. Agent executes target + noise (RNGDet: Gaussian, scale not stated; RNGDet++: evenly distributed); labels stay clean. ROI 256 px at 0.6 m/px (RNGDet) or 128 px at 1 m/px (RNGDet++). Seeds: local peaks of the intersection map I (RNGDet++: of a global heat map merged over ROIs), plus vertices created at junctions and candidates met on the way. Queries 5 / 10 / 20: APLS 62.78 / 67.88 / 65.31 (RNGDet Table III). Repo defaults (RNGDet++ README, code): move-ahead 30 px, sampling noise 8 px, validity threshold 0.75, seed peak threshold 0.7, candidate filter 30 px, merge ("alignment") distance 5 px. Step length is regressed, not fixed. Training script (`cityscale/main_train.py`): `ROI_SIZE` default 256 (the RNGDet++ paper text says 128; the shell scripts may override), 10 queries, 6 + 6 transformer layers, hidden 256, AdamW lr 1e-4 (backbone 1e-5), DETR matcher costs class 1 / L1 5 / giou 2, loss = CE + 5 L1 + segmentation (+ instance) terms, validation F1 at 3 px on skeleton points; the file contains no DAgger or rollout code.

## Input

RGB ROI (+ S, I maps, + history raster 1 x L x L); `image_feat` + `bev_raster`-like history.

## Output

up to 10 vertices (x, y) + validity per step; graph assembled by a FIFO candidate buffer. No stop or junction head: M comes from thresholding validity.

## Training

behaviour cloning on about 300 K (RNGDet) / 400 K (RNGDet++) samples, 50 epochs, lr 1e-4, 4 x RTX 3090; regime `teacher_forcing` + noise; Hungarian-type matching. DAgger is named in the text and rejected for sampling cost.

## Differentiable?

no - validity threshold / M logic, FIFO control flow, bipartite matching, per-step re-cropping.

## Sensors and data

aerial RGB. RoadTracer 40-city set (0.6 m/px, 4096 x 4096 tiles); City-scale (180 tiles, 2048 x 2048, 1 m/px, 144 / 9 / 27); SpaceNet (2,551 tiles of 400 x 400, 1 m/px).

## Reported results

City-scale TOPO-F1 / APLS: RNGDet++ 78.44 / 67.76, RNGDet 76.87 / 65.75, Sat2Graph 76.26 / 63.14; SpaceNet: 82.51 / 67.73 vs RNGDet 81.13 / 65.61 (RNGDet++ paper tables). RNGDet R-101 APLS 67.88 on the RoadTracer set (Table I). RNGDet++ ablation (City-scale F1 / APLS; the two single-removal rows are the instance head and the multi-scale features, row assignment unclear): 77.94 / 66.36 and 78.01 / 66.94 vs full 78.44 / 67.76, i.e. small gains of 0.4 - 0.5 F1 and 0.8 - 1.4 APLS. DeH4R reproduces RNGDet++ as 78.44 / 67.72.

## Compute

parameters not stated; one backbone pass per step: all test images City-scale 2.68 h (RNGDet) vs 3.85 h (RNGDet++) in the paper; DeH4R measures 136 min for RNGDet++.

## Code

https://github.com/TonyXuQAQ/RNGDetPlusPlus

## Fit to our project

reuse_idea - effort S (label rule, two modes, queue logic, query-count ablation, instance-mask-ahead auxiliary target). adapt_code is not advised: per-step image ROI + backbone, bipartite matching.

## Needs

seed set from a heat map; history as raster or token path; simplified GT graph with BFS expert; K output slots with a validity score.

## Conflicts

Hungarian matching of K slots (our losses are matching-free); BC only, no gradient through the rollout; per-step crop vs cached pyramid (adapter: grid_sample of the f0 / f1 / f2 pyramid around v_t).

## Tried in our experiments

E94 - E99 (learned trace network, partial: F1@1 0.851 vs 0.867, seeds limited) is the same family; E31 / E32 (matched sparse DETR decoder dropped) used matching on the whole image, not per step.
