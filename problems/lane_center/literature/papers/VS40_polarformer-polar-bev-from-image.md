# VS40 - PolarFormer: polar BEV from image columns, multi-scale polar maps, polar reference points

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S2.*

## Papers

PolarFormer (Jiang et al., AAAI 2023) https://arxiv.org/abs/2206.15398

## Idea

Each camera sees a wedge, so the BEV is a (radius x azimuth) grid; every image column is attended into one polar ray by cross-attention; the decoder uses deformable attention with reference points in cylindrical coordinates.

## Formulation

multi-scale polar maps (radius, azimuth) = (64, 256), (32, 128), (16, 64); best single setting N_theta = 256, R = 64 (39.6 mAP / 45.8 NDS); azimuth phi = arctan((x_I - u0) / f_x), radius rho = z sqrt(((x_I - u0)/f_x)^2 + 1); 3 cross-plane layers per scale; alignment samples polar points into each camera with a binary visibility weight; 6 encoder + 6 decoder layers; offsets (d rho, d phi, d z). The authors' argument: Cartesian BEV downsamples near regions and upsamples far regions without adding information.

## Input

image_feat from six cameras. output: 3-D boxes.

## Training

hungarian_set (decoder).

## Differentiable?

partial.

## Sensors and data

nuScenes cameras.

## Reported results

val R101 39.6 mAP / 45.8 NDS; ablation: Cartesian / Cartesian 38.1 / 45.5, polar features + Cartesian prediction 38.5 / 45.0, polar / polar 39.6 / 45.8; single-scale polar 39.1 / 45.0; a loss computed in polar coordinates failed (31.5 / 38.2, attributed to the 0 / 2 pi discontinuity) so the loss stays Cartesian; fixed sinusoidal positional encoding best (39.6) vs learned 2-D 38.8.

## Compute

parameters not stated.

## Code

URL from memory, not opened: https://github.com/fudan-zvg/PolarFormer

## Fit to our project

not_suitable (image-to-polar encoder needs perspective features); reuse_idea - S - resolution decaying with range, and the lesson to compute losses in Cartesian metres.

## Needs

images.

## Conflicts

cached Cartesian raster.

## Tried in our experiments

none.
