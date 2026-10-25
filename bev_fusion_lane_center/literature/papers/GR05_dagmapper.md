# GR05 - DAGMapper (conv-RNN lane-boundary walker trained through a 5-step unroll with Chamfer loss)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (paper HTML, two passes); gradient detail through the RoI marked not stated.. Pipeline stage: S4, T, L.*

## Papers

DAGMapper: Learning to Map by Discovering Lane Topology (Homayounfar et al., ICCV 2019; arXiv 2020) https://arxiv.org/abs/2012.12377

## Idea

Lane boundaries of highway forks and merges are unrolled as a DAG. Each node holds position, turning direction and a state {continue, fork, terminate}. Three convolutional-RNN heads (direction, position, state) read a bilinearly interpolated, rotated RoI of a BEV feature map around the parent. One of only two growers here that are trained through several of their own steps (the other: Trexplorer Super): argmax replaced by softmax, truncated back-propagation through 5 steps, symmetric Chamfer loss on dense samples of the produced polylines.

## Formulation

p(G|D) = prod_i p(theta_i | .) p(x_i | theta_i, .) p(s_i | theta_i, .) (Eq. 3); inference = greedy argmax of theta, then x, then s. Loss L = 1 Chamfer + 100 cosine (direction) + 10 focal (state) + 10 l2 (distance transform, DT); the global network also outputs a thresholded inverse-DT image that is concatenated to the features and skeletonised for initial vertices (skeleton endpoints); boundaries missed by the graph are re-seeded from uncovered DT regions. Training: start from GT initial vertices with small Gaussian noise; GT states are fed to the heads; predicted direction is used unless the parent is > 30 px from the GT boundary (then GT direction); rollouts stop at the image edge, a GT terminate, or 30 steps; BPTT 5 steps; AMSGrad lr 1e-3, 8 GPUs, 12 h. Fork: the parent re-enters the queue to spawn the second child; merge = terminate (no explicit merge target). RoI size and test-time step cap not stated; step size comes from the position softmax inside the RoI.

## Input

aggregated lidar-intensity BEV image, 5 cm / px, crops up to 8000 x 1200 px; `bev_raster` features (FPN, C channels not stated).

## Output

DAG of boundary points with states.

## Training

`teacher_forcing` partly (GT states, GT direction guard) + free rollout of predicted positions with `matching_free` Chamfer loss; softmax relaxation.

## Differentiable?

partial - arg max replaced by softmax in training only; whether gradients pass the RoI rotation / bilinear sampling into the direction head is not stated; test time is argmax.

## Sensors and data

lidar intensity only; 3,000 images from 68 fork / merge areas on one North American highway (70 / 15 / 15 split by area), generalisation set 336 images from another highway.

## Reported results

F1 at 2 / 3 / 5 / 10 px (5 cm px): 76.3 / 88.8 / 94.4 / 96.3; the DT baseline gives 66.9 / 86.4 / 94.9 / 97.0 (better at 5 and 10 px); 89 % of GT boundaries with correct topology; new highway F1@3px 69.0 (Tables 1 - 3). Ablations F1@2px (Table 2): adding the conv-RNN memory 69.5 -> 76.0 (without DT recovery) and 73.1 -> 76.3 (with it, = full model); no state head 70.0 vs 76.3 (full).

## Compute

not stated beyond training time; one 1200 x 8000 image per GPU.

## Code

none found

## Fit to our project

reuse_idea - effort M: the *training recipe* (short unroll, softmax relaxation of the selection, Chamfer rollout loss, GT guard against runaway states) is what our free-rollout, matching-free tree needs; fork / merge states per node are a simple event head (we have continue / end / fork / merge).

## Needs

a feature map the RoI can be sampled from; an initial-vertex source.

## Conflicts

the RoI is rotated by the *predicted* direction, so position loss depends on direction error (coupling); single child per step plus a fork queue.

## Tried in our experiments

E94 trace network (events continue / end / fork / merge) is close; no BPTT variant tried (E121 sent a planner loss through positions: no gain). E44 / E53 (learned split / merge key points, Bezier connectors) lifted split / merge F1 from 0.34 / 0.38 to 0.55 / 0.61 and 0.71 / 0.64 in system A, in line with DAGMapper's state-head ablation (about -6 F1@2px without it).
