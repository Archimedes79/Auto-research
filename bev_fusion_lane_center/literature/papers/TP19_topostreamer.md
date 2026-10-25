# TP19 - TopoStreamer (streaming query propagation for lane segments)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S8 L.*

## Papers

TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving (Yang et al., arXiv 2507.00709) https://arxiv.org/abs/2507.00709

## Idea

Previous-frame top-k queries are mapped to the current frame by an ego-motion MLP and replace the lowest-confidence queries; supervision of these stream queries uses instance IDs for the next frame's targets; a dynamic boundary positional encoding and lane denoising are added.

## Formulation

BEV fusion F~ = GRU(Warp(F_{t-1}, Psi), F_t); Q_t^S = MLP(concat(Q_{t-1}, Psi)) + Q_{t-1}; 200 queries of which 30 % (66) are stream queries replacing the lowest-confidence ones in layer 1; DN queries 240 (not propagated). Stream losses on centerline coordinates, classes, boundary offsets and BEV mask (kappa weights 1.0, 0.01, 1.0, 1.0, 0.025, 1.0, 3.0); matched stream queries tracked by OpenLane-V2 instance IDs ("lossless supervision"); adjacency from predecessor / successor MLPs, threshold 0.5. DBPE: sinusoidal PE of 8 boundary reference points recomputed per layer. DN: box shift factor 0.2, label flip 50 %, topology DN focal.

## Input

BEV 200 x 100 (+-50 x +-25 m), 7 cameras 2048 x 1550.

## Output

lane segments + adjacency.

## Training

Hungarian; first 12 epochs single-frame; 24 epochs; AdamW 2e-4; V100.

## Differentiable?

partial - Hungarian, top-k selection of stream queries.

## Sensors and data

camera; OpenLane-V2 subset A (subset B via pseudo-segments from centerlines + 4 m width).

## Reported results

module ablation (mAP / TOP_lsls / Acc_b): LaneSegNet 32.6 / 25.4 / 45.9; +DBPE +DN 34.5 / 26.7 / 47.6; +stream + DBPE 35.6 / 27.8 / 49.5; full 36.6 / 28.5 / 50.0; without ID tracking 34.4 / 28.0 / 49.5. Stream fraction 10 / 30 / 50 / 75 %: mAP 34.8 / 36.6 / 34.3 / 32.9. StreamMapNet 20.3 mAP, SQD-MapNet 26.0 (their numbers). Subset B centerline OLS 42.6 vs LaneSegNet 38.7.

## Compute

13.6 FPS vs 14.7.

## Code

https://github.com/YimingYang23/TopoStreamer

## Fit to our project

reuse_idea - effort S - our time recursion already warps the previous set; the numbers say the propagated share matters (30 % optimum) and that instance identity gives cleaner supervision.

## Needs

instance IDs or mask matching for stream supervision.

## Conflicts

assumes known relative pose; slower.

## Tried in our experiments

E39 / E41 (temporal feature fusion +0.008 F1): consistent with small gains on a strong single-frame result.
