# VS31 - LaneSegNet: lane attention (reference points along the segment, identical initialisation, heads-to-regions)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S5, S6.*

## Papers

LaneSegNet (Li et al., ICLR 2024) https://arxiv.org/abs/2312.16108

## Idea

Deformable attention whose reference points lie uniformly along the predicted lane segment (4 per boundary = 8) and whose different heads attend different regions of the segment; in the first decoder layer all heads share the SAME reference point from the positional query (identical initialisation).

## Formulation

LaneAttn(q_i, p_i, B) = sum_m W_m [sum_k a_imk W'_m Bilinear(B, p_i,m + dp_imk)]; offsets from a linear layer of the query, initialised so that K = 32 sampling locations lie evenly around each reference point (4 distances x 8 directions, as in Deformable DETR); single-scale BEV (200 x 100 cells over +-50 m x +-25 m = 0.5 m cells); 6 decoder layers (self-attention 8 heads, lane attention 8 heads, FFN 512); 200 queries of 512 dims (256 reference part + 256 instance part); lane segment = centreline + two boundaries, 10 points each; losses: L1 0.025, mask CE 3.0 + dice 1.0, focal class 1.5, line type 0.01, topology focal 5.0; Hungarian (or Sinkhorn); AdamW 2e-4, 24 epochs, 8 V100.

## Input

bev_raster. output: polyline segments + adjacency.

## Training

hungarian_set.

## Differentiable?

partial (matching).

## Sensors and data

OpenLane-V2 camera BEV (the paper says nuScenes lacks lane-segment labels, so it is not used).

## Reported results

ablation (mAP / TOP_lsls): deformable attention one reference point 28.7 / 6.9; hierarchical queries 29.1 / 5.2; multi-point attention 24.6 / 6.9; lane attention 32.6 / 8.1; without heads-to-regions and identical init 28.8 / 5.6; full model 32.6 mAP at 14.7 FPS (tiny 28.5 at 16.2 FPS); centreline reshaped DET_l 31.8 vs TopoNet 24.9; latency 20.67 ms vs 30.14 ms for a multi-branch baseline.

## Compute

above.

## Code

URL from memory, not opened: https://github.com/OpenDriveLab/LaneSegNet

## Fit to our project

reuse_idea / reimplement - M - closest published BEV-grid read: bilinear reads on a 0.5 m BEV raster around reference points on the hypothesis; adding reference points indiscriminately hurt (24.6), the head-to-region split is the gain.

## Needs

BEV raster (ours), Hungarian for the decoder as published.

## Conflicts

Hungarian; mmcv / mmdet3d codebase (not in our env).

## Tried in our experiments

E31 / E32 dropped; E17 / E18 transformer decoder -0.11.
