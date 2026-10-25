# GR14 - RNTR (RoadNet sequences: autoregressive, semi-autoregressive and non-autoregressive decoding)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper HTML up to appendix; README).. Pipeline stage: S4, S6, E (landmark / reachability metrics), B (TIT).*

## Papers

Translating Images to Road Network: A Sequence-to-Sequence Perspective (Lu et al., ICCV 2023; TPAMI 2025; arXiv 2402.08207) https://arxiv.org/abs/2402.08207

## Idea

The road network (a DAG of landmarks: crossroads, stop lines, fork and merge points) is serialised as an integer sequence; three decoders: AR (token by token), SAR (a DETR key-point decoder first, then sub-trees rooted at key points decoded in parallel, 6 x 18 tokens), NAR (SAR fine-tuned with masked-LM at 90 % mask, 3 mask-predict iterations). Also introduces landmark and reachability metrics and "topology-inherited training" (distilling camera BEV towards lidar-based features).

## Formulation

Graph -> forest: each merge point gets clones of its parents except the first; DFS order; six integers per vertex-edge pair: x, y, category (Ancestor / Lineal / Offshoot / Clone), parent index, two Bezier control offsets (+10); vocabulary 576 (coordinates 0 - 199, category 200 - 249, parent 250 - 349, control 350 - 569, special 570 - 573). AR loss = class-weighted MLE; SAR key-point loss = Hungarian (class + L1); NAR = masked-token likelihood. Sequence augmentation hurt; sequence noise padding (noise vertices with a noise class) is kept because AR models stop early. DFS tie-break: random or coordinate-based (numbers not stated). Training 300 epochs, lr 2e-4, flip / rotation / scaling of BEV features.

## Input

LSS BEV (128 x 352 images, BEV x in [-48, 48], y in [-32, 32] m), ResNet-50 or VoVNetV2; pre-trained BEV encoder frozen; `image_feat` -> `token_sequence`.

## Output

integer sequence -> `graph`.

## Training

`next_token` (AR), `hungarian_set` (SAR key points), masked-LM (NAR).

## Differentiable?

no - tokenisation, argmax / mask-predict, Hungarian key points.

## Sensors and data

6 cameras; nuScenes (default and a non-overlapping split), Argoverse; optional SD map (SD-RNTR).

## Reported results

nuScenes multi-camera default split landmark-F / reachability-F: AR 54.1 / 61.3 at 0.1 FPS, SAR 56.0 / 65.7 at 0.6 FPS (6 x), NAR 55.1 / 65.1 at 5.5 FPS (47 x); non-overlapping split AR 36.5 / 40.9, SAR 36.9 / 43.0; with TIT AR 39.1 / 49.8 (Tables IV - V). Mask ratio 90 %: 1 / 3 / 6 iterations 59.0 / 60.2 / 60.3 landmark-F (VoVNetV2 row). Removing intra-sequence attention drops reachability-F from 66.0 to 56.3.

## Compute

AR 37.25 M parameters, NAR 35.90 M; NAR about 6 GB at batch 2.

## Code

https://github.com/fudan-zvg/RoadNet

## Fit to our project

reuse_idea - effort S: SAR = a *fixed-shape set of sub-trees rooted at key points* decoded in parallel with intra / inter-sequence attention is the closest published structure to our fixed-shape tree; the metrics are reusable.

## Needs

key-point decoder (matching); sequence order; BEV encoder.

## Conflicts

tokens and Hungarian key points.

## Tried in our experiments

none.
