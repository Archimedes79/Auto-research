# PL16 - Direct BEV-feature attention for behaviour prediction (cached mapping features into the predictor)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, B.*

## Papers

Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention (Gu et al., ECCV 2024) https://arxiv.org/abs/2407.06683

## Idea

Skip map decoding; the predictor reads the mapping model's BEV feature grid. HiVT: the BEV grid is split into flattened patches projected to N tokens, agent patches are queries, all map patches keys / values, vector lanes removed. DenseTNT: BEV features refined by a 1-D CNN and sampled at each lane vertex, concatenated with vertex positions (hidden size doubled), or agent subgraphs replaced by agent-BEV attention.

## Formulation

BEV grid 200 x 100 for MapTR / MapTRv2 and 100 x 50 for StreamMapNet over 60 x 30 m; patch sizes (20, 10), (20, 20) for the centre-line model, (10, 5); attention MLP dims 512 (MapTR), 64 (MapTRv2, CL), 128 (StreamMapNet). Features are extracted once and stored (the pre-built datasets are about 500 GB each and contain the BEV features).

## Input

stored BEV features, agent histories (and lanes for DenseTNT).

## Output

agent trajectories.

## Training

mapping models trained to convergence first, predictors trained on the stored features; joint training listed as future work.

## Differentiable?

yes inside each stage; no gradient from prediction to mapping.

## Sensors and data

nuScenes, MapTR / MapTRv2 / StreamMapNet cameras.

## Reported results

minADE / minFDE / MR vectorised -> BEV: HiVT + MapTR 0.4234 / 0.8900 / 0.0955 -> 0.3617 / 0.7401 / 0.0720; DenseTNT + MapTR 1.0462 / 2.0661 / 0.3494 -> 0.7608 / 1.4700 / 0.2593; HiVT + MapTRv2-CL 0.3657 / 0.7473 / 0.0710 -> 0.3652 / 0.7323 / 0.0710 (no change). No comparison to the GT map.

## Compute

up to 73 % faster inference (MapTR + HiVT 22.4 -> 9.1 ms on an RTX 4090).

## Code

https://github.com/alfredgu001324/MapBEVPrediction

## Fit to our project

reuse_idea - effort S - this is our situation (frozen cached BEV features, consumer reads them): a planner can read the 200 x 200 pyramid at the vertex positions in addition to the decoded graph.

## Needs

BEV features aligned to the ego frame; vertex positions for sampling.

## Conflicts

staged training means the consumer does not shape the BEV model (as with our frozen foundation).

## Tried in our experiments

E117 reads edges + route probability + ego motion; reading BEV features at vertices is not listed as tried.
