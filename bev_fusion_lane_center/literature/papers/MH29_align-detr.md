# MH29 - Align-DETR (IoU-aware BCE target for the score; rank-weighted k positives)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L.*

## Papers

Align-DETR: Enhancing End-to-end Object Detection with Aligned Loss (Cai et al., 2023) https://arxiv.org/abs/2304.07527 (read via ar5iv)

## Idea

Train the classification score toward a joint quality metric so that confidence reflects localisation quality (ranking by score then equals ranking by quality); positives use a soft target, negatives focal loss; intermediate layers take k = 4 matched candidates per GT with rank-decaying weights.

## Formulation

target t_c = exp(-r / tau) * q, q = p^alpha * u^(1 - alpha), r = candidate rank, alpha = 0.25, tau = 1.5; regression weight exp(-r / tau); k = 4 in intermediate layers, k = 1 in the last. - input / output: DETR decoders.

## Training

COCO 12 / 24 epochs; regime `hungarian_set` (modified).

## Differentiable?

yes.

## Sensors and data

COCO.

## Reported results

Align-DETR 50.5 AP (12 ep) / 51.7 (24 ep); DINO 1x 49.0; H-DETR baseline 48.7 -> 49.3. The paper does not frame it as calibration.

## Compute

not stated.

## Code

not checked

## Fit to our project

reuse_idea - effort S - template for the validity score target: p^alpha * closeness^(1 - alpha) with closeness from point-to-polyline distance.

## Needs

a closeness measure in [0, 1].

## Conflicts

none.

## Tried in our experiments

none
