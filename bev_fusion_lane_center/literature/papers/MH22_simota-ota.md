# MH22 - SimOTA / OTA (dynamic-k positives per ground truth, no Hungarian)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M (limited)** (A abstract, M method, C code page). Verified: partly. Pipeline stage: L.*

## Papers

YOLOX: Exceeding YOLO Series in 2021 (Ge et al., 2021) https://arxiv.org/abs/2107.08430 (ar5iv, SimOTA described briefly; the dynamic-k rule itself is in the OTA paper, not read)

## Idea

Positives per GT are the k lowest-cost predictions inside a centre region, with k varying per GT; SimOTA replaces the Sinkhorn-Knopp solver of OTA by this dynamic top-k.

## Formulation

c_ij = L_cls + lambda * L_reg (lambda value not given on the page); the page does not give the rule for k (refers to OTA), the conflict rule or the centre radius. - input / output: dense detector heads; boxes.

## Training

regime `matching_free` (many-to-one by dynamic k).

## Differentiable?

yes (labels constant).

## Sensors and data

COCO.

## Reported results

"+multi positives" baseline 45.0 AP -> SimOTA 47.3 AP (+2.3); OTA Sinkhorn-Knopp adds about 25 % training time.

## Compute

SimOTA cheaper than OTA, no number.

## Code

not checked

## Fit to our project

reuse_idea - effort S - a variable number of positive children per tip depending on quality.

## Needs

cost per pair.

## Conflicts

needs NMS at inference (many positives).

## Tried in our experiments

none
