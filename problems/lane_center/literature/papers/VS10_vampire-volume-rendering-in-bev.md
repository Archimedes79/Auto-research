# VS10 - Vampire: volume rendering in BEV along the height axis

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, B, L.*

## Papers

Regulating intermediate 3D features for vision-centric autonomous driving ("Vampire", arXiv 2312.11837; authors not captured) https://arxiv.org/abs/2312.11837

## Idea

Besides rendering depth / semantics in camera view, a BEV feature volume is rendered "directly from the top-down height axis" into BEV height and BEV semantic maps; these rendering losses regularise intermediate features during joint training.

## Formulation

density V_density = alpha * Psi_beta(V_sdf), Psi_beta = Laplace CDF with learnable alpha, beta > 0; T(t) = exp(-integral sigma) and w(t) = integral T sigma as printed; height D(r) = integral w r, semantics S(r) = integral w s; BEV volume 20 x 256 x 256 over height [-3, 5] m; maps rendered at 1/4 resolution; losses smooth-L1 (height / depth), CE + Lovasz (semantics), both weights 1; number of samples along the vertical axis not stated.

## Input

bev_raster volume with a height axis. output: BEV height and semantic maps.

## Training

auxiliary rendering losses in joint training (the paper does not call it pre-training); matching_free.

## Differentiable?

yes.

## Sensors and data

nuScenes cameras (lidar labels for depth / height).

## Reported results

nuScenes val ResNet-50, no temporal fusion: detection mAP 0.301, NDS 0.354 (BEVFormer 0.257 / 0.359 and BEVDet 0.286 / 0.372 as cited from Table 3); Occ3D-nuScenes mIoU 28.3 (appendix 28.33; the +1.4 vs BEVFormer in the text is 1.45 in the appendix); lidar segmentation mIoU 66.4 vs 59.3 (TPVFormer).

## Compute

not stated.

## Code

github.com/cskkxjk/Vampire (named in the abstract; release "upon publication" per appendix; not opened).

## Fit to our project

not_suitable (needs a trainable 3D / BEV volume); reuse_idea - S - the only BEV-plane example found: rays can be cast through a BEV feature volume, but along height, not along a lane.

## Needs

height-resolved volume. compat_conflicts: our cached features have no height axis.

## Tried in our experiments

none.
