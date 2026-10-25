# SN02 - RNTR: frozen BEV encoder + sequence head (closest precedent for a frozen interface)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (first ~100k of ~117k characters read)** (A abstract, M method, C code page). Verified: partly (reader summary; part of the page not read). Pipeline stage: B, T, D.*

## Papers

RNTR, road network from surround cameras as an integer sequence (Lu et al., ICCV 2023; title from memory) https://arxiv.org/abs/2402.08207

## Idea

A camera BEV encoder (LSS-style) is first pretrained for centerline segmentation, then frozen while a sequence transformer (autoregressive, semi-AR or non-AR) is trained on its features. A later "topology-inherited training" (TIT) trains the transformer on clean lidar-BEV features, distils the camera encoder to those features (L1) and fine-tunes everything jointly. Unfreezing the encoder barely matters; the split choice matters a lot.

## Formulation

BEV x -48..48 m, y -32..32 m (cell size written "11m", probably a typo for 1 m; not confirmed); images 128 x 352; ResNet-50 (Cityscapes DeepLabV3+ pretrained) or VoVNetV2; TIT: stage 1 transformer on lidar-BEV features (300 epochs), stage 2 camera encoder distilled by L1 to the frozen lidar-BEV features (24 epochs), stage 3 joint fine-tune (100 epochs). Token layout and losses not read in detail.

## Input

surround images (+ optional OpenStreetMap SD-map sequence as prompt); BEV channel count not stated.

## Output

token sequence (vertices, parent index, Bezier control points).

## Training

regime next_token (AR) / parallel decoding (NAR); encoder frozen in the default setup (they "load and freeze the parameters of the BEV encoder").

## Differentiable?

partial - token sampling / argmax at inference; encoder frozen.

## Sensors and data

camera (lidar only in TIT stage 1, SD map optional), nuScenes, 96 x 64 m BEV.

## Reported results

default (overlapping) split landmark F / reachability F: AR 54.1 / 61.3, semi-AR 56.0 / 65.7, NAR 55.1 / 65.1 (Table V). Non-overlapping split: 36.5 / 40.9, 36.9 / 43.0, 36.2 / 42.3; AR + TIT 39.1 / 49.8; SD-map variant + TIT 54.7 / 62.6. Frozen vs unfrozen BEV encoder (assembled stage, Table X): landmark F 38.6 vs 39.1, reachability F 48.6 vs 49.8 (the page calls the difference not significant).

## Compute

NAR 5.5 FPS vs AR 0.1 FPS (hardware not read).

## Code

not checked.

## Fit to our project

reuse_idea - effort S - same regime as ours (frozen BEV encoder, head trained on its features, evaluation on an overlap-free split); the token head itself is out of scope (sampling, argmax).

## Needs

a BEV feature map from a segmentation-pretrained encoder; an overlap-free split for honest numbers.

## Conflicts

token sequences need argmax / sampling (not differentiable); the lidar-BEV teacher in TIT needs lidar (not available to us).

## Tried in our experiments

none for the sequence head; the frozen-foundation regime itself is our system B.
