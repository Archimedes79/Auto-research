# SN33 - Encoder-side temporal fusion versus our set-level recursion (summary of seen numbers)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (from the cards)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S8, B.*

## Papers

BEVFormer SN22 ; StreamMapNet SN18 ; HDMapNet SN15 ; CRN SN05 ; BEVDepth SN24 ; Simple-BEV SN06 ; GaussianCaR SN01 ; CenterLineDet SN19

## Idea

Encoder-side temporal fusion aligns the previous BEV (or several) by ego motion and fuses by attention, GRU or concatenation; gains are largest for velocity and occluded objects, modest for static map geometry. GaussianCaR has no temporal camera state; its only temporal input is 7 radar sweeps.

## Formulation

BEVFormer: recurrent temporal self-attention over ego-aligned B_(t-1); StreamMapNet: warp + GRU + LayerNorm, no gradient to previous frames; CRN: aligned fused BEV maps concatenated; CenterLineDet: warped neighbour heat maps averaged.

## Input

previous BEV(s) + ego pose.

## Output

fused BEV.

## Training

end to end with the encoder.

## Differentiable?

yes.

## Sensors and data

nuScenes.

## Reported results

BEVFormer-S val NDS by frames 0.448 / 0.490 / 0.510 / 0.517 (1 - 4); HDMapNet map IoU 32.9 / 35.8 / 36.4 (1 / 2 / 4 frames); StreamMapNet AV2 ablation: BEV fusion +2.4 mAP; CRN mAP 42.9 -> 48.4 (1 -> 5 frames, orientation error unstable); BEVDepth NDS 0.367 -> 0.442 with multi-frame; GaussianCaR radar sweeps 54.9 / 56.2 / 57.3 vehicle IoU for 1 / 4 / 7.

## Compute

n/a.

## Code

n/a.

## Fit to our project

reuse_idea - effort S - our time recursion works on vertices with ego-warped validity; encoder-side fusion cannot be added to a frozen cache (needs the encoder), only head-side (input memory = ego-warped previous input; E39 / E41: +0.001 / +0.008 F1).

## Needs

ego poses.

## Conflicts

encoder-side fusion needs retraining the foundation.

## Tried in our experiments

E39 / E41 (temporal feature fusion on the cache).
