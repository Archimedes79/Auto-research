# VS42 - Pyramid level assignment by region size (FPN)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2.*

## Papers

Feature pyramid networks for object detection (Lin et al., CVPR 2017) https://arxiv.org/abs/1612.03144

## Idea

Which pyramid level a region reads from is a function of its size: RoIs of area w * h use level k = floor(k0 + log2(sqrt(w h) / 224)) with k0 = 4 (P2..P5; P6 only for RPN anchors).

## Formulation

top-down pathway with nearest-neighbour 2x upsampling, 1x1 lateral conv, element-wise add, 3x3 smoothing conv, 256 channels; RPN anchors one scale per level 32^2..512^2, three ratios.

## Input

backbone pyramid. output: feature pyramid.

## Training

standard detection; none matching.

## Differentiable?

yes.

## Sensors and data

images (COCO).

## Reported results

COCO minival ResNet-50: RPN AR1k: C4 baseline 48.3, finest level only 51.3, full pyramid 56.3; Fast R-CNN AP 31.9 / 33.4 / 33.9; top-down without lateral connections 46.1 AR1k; bottom-up pyramid only 49.5 (24.9 AP).

## Compute

not stated.

## Code

not checked.

## Fit to our project

reuse_idea - S - a size-to-level rule: replace object size by ring spacing or tip uncertainty to decide how much of f0 / f1 / f2 to read at ring k (own mapping; no paper does this by DISTANCE FROM THE EGO for centre-lines).

## Needs

pyramid.

## Conflicts

none.

## Tried in our experiments

E94 uses the pyramid f0 / f1 / f2 (fixed assignment).
