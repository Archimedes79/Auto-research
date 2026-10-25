# VM21 - MambaMap (state-space memory)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8.*

## Papers

MambaMap: Online Vectorized HD Map Construction using State Space Model (arXiv 2507.20224; authors not read) https://arxiv.org/abs/2507.20224

## Idea

A sliding memory bank of the last N = 4 frames holds refined BEV features and refined instance queries (detached); history BEV is warped by a 4x4 affine, history queries are reordered to the current ones by Hungarian matching on L2 distance between query vectors; a gated diagonal-state-space block scans the BEV (50 x 100 = 5000 tokens, four directions) and the instance sequences (length (N+1) Nq).

## Formulation

best variant DSS (Mamba and Mamba-2 scored lower); N = 6 gave 66.5 vs 67.3 at N = 4; loss 5 x smooth-L1 points + 50 x focal; Deformable-DETR-style decoder, Hungarian as StreamMapNet; two-stage training (single frame, then random sequences); R50 + BEVFormer, lr 2.5e-4, batch 8, 2 x A30, 30 / 24 epochs.

## Input

BEV + queries of N + 1 frames.

## Output

polylines.

## Training

hungarian_set.

## Differentiable?

partial.

## Sensors and data

camera; nuScenes, AV2; 60 x 30 and 100 x 50 m.

## Reported results

nuScenes 60 x 30: 67.3 (12.7 FPS) vs StreamMapNet 63.4, SQD-MapNet 65.8; 100 x 50: 66.5; new split 30 m: 40.1 vs StreamMapNet 34.1, SQD 37.3, MemFusionMap 38.0; AV2 new split 61.0.

## Compute

parameters not reported.

## Code

announced https://github.com/ZiziAmy/MambaMap (may not be live)

## Fit to our project

reuse_idea - M - low priority; an SSM memory is an alternative to the GRU, with evidence only from this paper.

## Needs

BEV sequence. Cached-BEV: yes.

## Conflicts

query reorder by Hungarian.

## Tried in our experiments

E41 (GRU-type fusion).
