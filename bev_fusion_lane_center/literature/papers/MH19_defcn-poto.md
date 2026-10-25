# MH19 - DeFCN / POTO (prediction-aware one-to-one assignment without Hungarian, 3D max filtering)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, S4.*

## Papers

End-to-End Object Detection with Fully Convolutional Network (Wang et al., CVPR 2021) https://arxiv.org/abs/2012.03544 (read via ar5iv)

## Idea

In a fully convolutional detector, replace the many positives per object by exactly one positive chosen by a prediction-aware quality (class score and IoU, within a spatial prior); add a 3D max filter over neighbouring locations and scales so duplicates are suppressed inside the network; use an auxiliary one-to-many loss for dense supervision. NMS is not needed.

## Formulation

Q = 1[pred in Omega_i] * p^(1 - alpha) * IoU^alpha, alpha = 0.8; Omega_i = FCOS-style centre sampling; multiplicative fusion beats additive by 0.7 mAP; alpha = 0 or 1 close the NMS gap less. Selection is an argmax over assignments of the summed Q (the page could not confirm per-GT argmax vs joint). 3D max filter: resample features from tau = 2 neighbouring scales to the current size, then max over scales and a 3 x 3 neighbourhood per channel, GroupNorm + sigmoid. Auxiliary loss: top-9 candidates per FPN level by Q, positives above mean + std of Q, focal loss; training only. - input / output: FPN feature maps -> box + score maps.

## Training

FCOS / Detectron2 2x recipe; regime `matching_free` (quality-based one-to-one) + auxiliary one-to-many.

## Differentiable?

yes (assignment is a constant label; the max filter passes gradient to the max).

## Sensors and data

COCO, CrowdHuman (images).

## Reported results

COCO val R50 without NMS: POTO 38.0; + 3DMF 39.8; + aux 41.1 (with NMS 41.2); FCOS without NMS 19.0, with NMS 40.9; removing NMS from one-to-many FCOS drops 40.5 -> 12.1. CrowdHuman AP50 89.1 vs 86.1.

## Compute

3DMF adds little; 36 epochs vs 500 for DETR.

## Code

https://github.com/Megvii-BaseDetection/DeFCN

## Fit to our project

reuse_idea - effort S - class probability x geometric quality, argmax per GT, is a matching-free rule for "which ray sample is the positive"; 3D max filter = learned NMS over ray and ring neighbours.

## Needs

dense score maps; a geometric quality (IoU analogue: distance).

## Conflicts

labels depend on current predictions (non-stationary); code is Detectron2-based.

## Tried in our experiments

none
