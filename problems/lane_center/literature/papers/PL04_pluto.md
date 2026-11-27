# PL04 - PLUTO (route-restricted reference lines as lateral queries, contrastive negatives)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, S4, T.*

## Papers

PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving (Cheng, Chen, Chen; 2024) https://arxiv.org/abs/2404.14327

## Idea

One trajectory per (reference line x longitudinal query). Reference lines come from the lane graph and the route; the 12 longitudinal queries are learned embeddings. An auxiliary drivable-area loss and a contrastive imitation loss with positive / negative scene augmentations are added; a reference-free head covers scenes without a reference line.

## Formulation

map polylines with 8 channels per point (p_i - p_0, p_i - p_{i-1}, p_i - p_left, p_i - p_right), PointNet encoder; speed limit and traffic-light status as attribute embeddings; Fourier embedding of global position. Reference lines (code, RouteManager): start lanes within 3 m of the ego on a route roadblock with heading error below 90 deg; depth-first search over `outgoing_edges` restricted to route roadblocks, stop at 15 lanes or 100 m; trim from the ego projection; drop paths shorter than 0.8 x length; de-duplicate when the maximum coordinate difference is below 0.5 m; feature = 100 points per line (every 4th sample) + future projection of the ego. Decoder: lateral queries (reference lines) x N_L = 12 longitudinal queries, D = 128, 4 + 4 layers; output 80 steps x 6 channels; top K = 20 of N_R x N_L modes. Loss = L_i (smooth-L1 on the target query selected by teacher forcing from the GT endpoint + CE on mode scores) + L_p + L_aux (hinge on an ESDF cost map 500 x 500 at 0.2 m, 3 covering circles) + L_c (softmax triplet, temperature 0.1), all weights 1. Positive augmentation: ego-state perturbation, dropout of non-interactive agents; negative: remove leading agent, insert a leading agent from another sample, remove interactive agent, invert traffic lights; batch tripled. Model code (pluto_model.py): defaults dim 128, 4 encoder + 4 decoder layers, 8 heads, 6 modes, drop-path 0.2, dropout 0.1, `use_ego_history=False`, state dropout 0.75, `ref_free_traj=False` (the reference-free head is optional; the evaluation fallback for scenes without reference line needs it enabled); the map encoder uses lane boundaries (`use_lane_boundary=True`); position embedding = Fourier embedding of (x, y, heading) of agents and polygon centres; key padding from the validity masks of agents, polygons and static objects; the reference-line validity only masks the output probabilities at evaluation.

## Input

map polylines, agents (20 steps), static obstacles, reference lines.

## Output

up to 20 scored trajectories.

## Training

loss above; 1 M frames, 25 epochs, batch 128, AdamW; regime: imitation with teacher-forced reference-line choice + contrastive.

## Differentiable?

yes for the network; reference-line generation is a rule (DFS), no gradient to the map.

## Sensors and data

nuPlan GT map.

## Reported results

Val14 non-reactive / reactive score: PLUTO with post-processing 93.21 / 92.06, without 89.04 / 80.01; PDM-Closed 93.08 / 93.20; PlanTF 85.30 / 77.07; PlanTF-H 89.96 / 88.08; GameFormer 82.95 / 83.88; log replay 93.68 / 81.24.

## Compute

4 x RTX 3090, about 45 h with the contrastive loss (22 h without); parameters not stated.

## Code

https://github.com/jchengai/pluto (README: training part not fully tested)

## Fit to our project

reuse_idea - effort M - DFS over route-filtered successors is exactly what our edge graph + route probability supports; the reference-free head is a model for the "no usable lane" case.

## Needs

successor edges, a route filter, polylines resampled to a fixed point count, drivable-area raster for the auxiliary loss.

## Conflicts

roadblock-level route; nuPlan devkit; DFS on a graph with false forks multiplies reference lines (only the 0.5 m de-duplication limits their number).

## Tried in our experiments

none.
