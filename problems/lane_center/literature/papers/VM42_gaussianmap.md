# VM42 - GaussianMap (Gaussian primitives splatted to BEV)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: B.*

## Papers

GaussianMap: Learning Gaussian Representation for Multi-Sensor Online HD Map Construction (arXiv 2606.31177, Jun 2026; authors not read) https://arxiv.org/abs/2606.31177

## Idea

20,000 Gaussian primitives (mean, scale, rotation, opacity, feature) replace the dense BEV grid; they sample LSS camera (and lidar) BEV features, interact by self- and cross-attention and are splatted (mixture responsibilities at cell centres) into a BEV feature that feeds a MapQR + DAMap decoder.

## Formulation

M = 20,000 matched to 200 x 100 cells; init uniform means, fixed scales, identity rotation, opacity 0.5; 3-layer encoder, 16 reference points per Gaussian; loss = map (class, p2p, direction) + dense (depth, BEV seg, PV seg), lambdas not given; 6 layers, 100 queries x 20 points; lr 6e-4, batch 32, 24 / 6 epochs.

## Input

sensor features.

## Output

polylines.

## Training

not stated.

## Differentiable?

yes in the splatting.

## Sensors and data

camera, lidar; nuScenes, AV2 standard val; no radar; no temporal part.

## Reported results

nuScenes val camera 70.5 mAP (MapQR + DAMap 68.8, MapTRv2 61.5); camera + lidar 78.3 (75.2, 69.0); AV2 camera 6 ep 70.1, C + L 79.0.

## Compute

FPS and parameters not reported.

## Code

none ("will be made publicly available").

## Fit to our project

not_suitable - M - an encoder-side representation; the splatting idea could turn sparse graph vertices into a dense feature but nothing else transfers.

## Needs

sensor encoders.

## Conflicts

replaces the BEV encoder.

## Tried in our experiments

none.
