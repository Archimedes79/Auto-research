# SN22 - BEVFormer: grid BEV queries, spatial cross-attention, recurrent temporal self-attention

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S8.*

## Papers

BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers (Li et al., ECCV 2022; from memory) https://arxiv.org/abs/2203.17270 ; code https://github.com/fundamentalvision/BEVFormer (URL from memory, not opened)

## Idea

Learnable grid BEV queries (200 x 200, C = 256, +-51.2 m, 0.512 m cells) attend to image features through deformable spatial cross-attention on 4 z-reference points per pillar (only cameras the points project into, the "hit" views, are used) and to the previous BEV (ego-motion aligned) through temporal self-attention; recurrent: only the previous BEV is carried.

## Formulation

6 encoder layers; N_ref = 4 points at heights -5 to 3 m; 4 sampling points per reference point; ResNet101-DCN / VoVNet-99; 24 epochs; 4 frames in training (3 past without gradient); detection by Deformable DETR (900 queries); segmentation by a Panoptic-SegFormer-style mask decoder.

## Input

6 images + previous BEV.

## Output

boxes, road / lane masks.

## Training

regime hungarian_set (detection), joint segmentation.

## Differentiable?

partial - Hungarian at training.

## Sensors and data

camera, nuScenes (200 x 200, +-51.2 m), Waymo.

## Reported results

val NDS 51.7 / mAP 41.6 (R101); temporal ablation (BEVFormer-S val NDS by frames): 1 frame 0.448, 2 0.490, 3 0.510, 4 0.517; map segmentation val (seg only): road 80.1, lane 25.7 (LSS 72.9 / 20.0); velocity error 0.802 -> 0.394; recall on the 0-40 % visibility subset +6.0 over the static variant. Latency 130 ms (V100, R101-DCN, 1600 x 900).

## Compute

130 ms default.

## Code

see papers line.

## Fit to our project

not_suitable - effort L - needs images and trainable queries; its hit-view test is reused in SN13 and its ego-aligned recurrent BEV is the encoder-side analogue of our time recursion (SN33).

## Needs

images; trainable queries.

## Conflicts

queries tied to the encoder.

## Tried in our experiments

none.
