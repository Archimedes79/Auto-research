# PL36 - VectorNet (vectors, polyline subgraphs, global attention without lane topology)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10.*

## Papers

VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation (Gao et al., CVPR 2020) https://arxiv.org/abs/2005.04259

## Idea

Every map element and trajectory is a polyline of vectors v_i = [start, end, attributes (type, speed limit, traffic-light state, timestamp), polyline id]; centre-line key points are sampled at equal spatial distance (spacing not stated). Polyline subgraph: 3 layers of shared FC (64 hidden) + layer norm + ReLU with concatenation of the max-pooled neighbourhood, final max-pool. Global graph: one self-attention layer over all polyline nodes with full connectivity (lane topology is not encoded).

## Formulation

node-completion auxiliary loss (masked nodes reconstructed, Huber, alpha 1.0, features l2-normalised); coordinates centred on the target agent's last position; outputs per-step offsets rotated to the target heading; trajectory loss negative Gaussian log-likelihood.

## Input

all polylines in the scene.

## Output

future trajectory of the target agent.

## Training

supervised; auxiliary masked-node reconstruction (self_supervised).

## Differentiable?

yes.

## Sensors and data

Argoverse (GT map) and in-house data.

## Reported results

Argoverse DE@3s: history only 5.24, map only 3.94, map + agents 3.84, + node completion 3.67; ConvNet (ResNet-18, 400 x 400) 4.49; test K = 1 DE@3s 4.01 vs 4.17 for the challenge winner.

## Compute

0.041 GFLOPs and 72 K parameters per target vs 10.56 GFLOPs and 246 K for the best ConvNet.

## Code

no official release mentioned on the page.

## Fit to our project

reuse_idea - effort S - the simplest polyline encoder, structurally robust to missing edges because it uses none.

## Needs

polylines + attributes.

## Conflicts

no topology information.

## Tried in our experiments

none.
