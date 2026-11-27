# VS16 - LETR: line segments as transformer queries (no sampling along the line)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S5.*

## Papers

Line segment detection using transformers without edges (Xu et al., CVPR 2021) https://arxiv.org/abs/2101.01909

## Idea

1000 line-entity queries regress two endpoints and a confidence; a coarse (C5, stride 32) then a fine (C4, stride 16) encoder-decoder refines them; nothing is sampled along the line.

## Formulation

each stage 6 encoder + 6 decoder layers, 256 channels, 8 heads; the fine decoder inherits the entities of the coarse one; Hungarian matching on lambda1 d(L^, L) - lambda2 p (lambdas not stated); focal-style adaptive BCE, L1 on endpoints (GIoU removed), deep supervision on all decoder layers with shared heads, background class weight reduced 10x; coarse 500 epochs, fine 325 (incl. 25 focal fine-tune), AdamW 1e-4, initialised from COCO DETR.

## Input

multi-level image features. output: 1000 scored segments.

## Training

hungarian_set.

## Differentiable?

partial (hungarian).

## Sensors and data

images.

## Reported results

Wireframe sAP10 65.2, sAP15 67.7, APH 86.3, FH 83.3 (ResNet-101); YorkUrban sAP10 29.4, APH 62.7, FH 66.9 (ResNet-50); 5.04 FPS on a V100.

## Compute

parameters not stated.

## Code

URL from memory, not opened: https://github.com/mlpc-ucsd/LETR

## Fit to our project

not_suitable - Hungarian matching and 1000 queries; only the coarse-to-fine inheritance is an idea.

## Needs

Hungarian. compat_conflicts: matching-free rule.

## Tried in our experiments

E31 / E32 matched sparse decoder dropped (grid +0.058 F1).
