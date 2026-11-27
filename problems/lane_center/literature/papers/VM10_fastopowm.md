# VM10 - FASTopoWM (fast-slow latent world model)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, T.*

## Papers

FASTopoWM (arXiv 2507.23325; title "Fast-Slow Lane Segment Topology Reasoning with Latent World Models" from memory) https://arxiv.org/abs/2507.23325

## Idea

Two streams share decoder weights: a fast single-frame stream and a slow stream with history through "world models" - a query world model and a BEV world model (2 transformer layers each, conditioned on the flattened relative pose) predict the NEXT frame's stream queries and BEV features, then fused with the current BEV by a GRU. At test time the slow output is used when the pose is reliable, the fast one otherwise.

## Formulation

L = L_slow + L_fast; L_slow = L_ls + 0.3 L_latent; L_latent = MSE(predicted stream BEV, current BEV) (self-supervised) + transformation loss on predicted queries (L1 coordinates, focal / CE class and type, CE + Dice mask) against previous-frame GT mapped to the current frame. About 30 % of the 200 queries are stream slots (inferred, K not stated). No denoising. The BEV world model runs on a 100 x 50 average-pooled BEV. Weights 0.025 / 3.0 / 1.5 / 0.01 / 5.0; 24 epochs (first 12 single-frame); AdamW 2e-4; batch 8.

## Input

as VM09 (200 x 100 BEV, +-50 x +-25 m, 7 cameras).

## Output

lane segments + adjacency.

## Training

Hungarian; regime hungarian_set + self_supervised (next-BEV MSE).

## Differentiable?

partial.

## Sensors and data

camera; OpenLane-V2 subsets A and B.

## Reported results

subset A lane segments mAP 37.4 (TOP_lsls 29.6, 11.4 FPS) vs its single-frame variant 34.1 and Topo2Seq 33.6; subset B centerline OLS 46.3, DET_l 35.1, TOP_ll 33.0 (single-frame 41.8 / 31.6 / 27.1; LaneSegNet 38.7 / 27.5 / 24.9; SQD-MapNet 29.1 / 21.9 / 13.3). Pose robustness not tested (future work).

## Compute

11.4 FPS (fast only 14.0).

## Code

https://github.com/YimingYang23/FASTopoWM

## Fit to our project

reuse_idea - M - (a) predict the next frame's BEV / elements from the previous ones given the pose, trained self-supervised against what the next frame shows (an MSE on cached features is possible for us), (b) a fast fallback when pose or history is unreliable - fits our synthetic pose-jump and failure tests.

## Needs

ego pose as a conditioning vector; a single-frame path of the same network. Cached-BEV: yes for the world-model part.

## Conflicts

two forward paths; Hungarian.

## Tried in our experiments

E41 (fusion, small); predictive next-frame memory: none.
