# PL42 - P-MapNet (OSM raster prior with cross-attention; HD-map prior as masked autoencoder)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S9.*

## Papers

P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors (Jiang et al., arXiv 2024) https://arxiv.org/abs/2403.10521 ; project https://jike5.github.io/P-MapNet/

## Idea

SD map (OSM road centre-line skeletons, weakly aligned because of GPS bias) rasterised into the ego frame; a conv encoder with sine position embedding gives SD tokens; BEV features (down-sampled by 4 to 50 x 25) cross-attend to them (2 layers best; 4 or more overfit). HD prior: ViT masked autoencoder (50 % mask, patch sizes 20 x 20 to 40 x 80) refines the initial segmentation; optional and heavy.

## Formulation

fusion comparison at 120 x 60 m (camera + lidar mIoU): none 49.07, concat 54.73, CNN concat 56.27, cross-attention 60.20.

## Input

BEV features, rasterised SD map, optional HD patches.

## Output

segmentation, then vectors by post-processing.

## Training

HD prior pre-trained 20 epochs, then fine-tuning 10 epochs.

## Differentiable?

yes (vectorisation is post-processing).

## Sensors and data

nuScenes, Argoverse 2.

## Reported results

nuScenes val camera + lidar mIoU: 60 x 30 m 44.40 -> 51.93 (SD) -> 53.07 (SD + HD); 120 x 60 m 49.50 -> 60.20 -> 61.77; 240 x 60 m 36.47 -> 48.87 -> 49.93; camera only 240 x 60 m 26.77 -> 42.20 -> 45.50; MapTR + SD at 240 x 60: mAP 8.03 -> 16.53. Including OSM service roads lowered mIoU 60.20 -> 58.53; no explicit GPS-noise experiment.

## Compute

not read.

## Code

project page (code announced there)

## Fit to our project

reuse_idea - effort S - cross-attention to a rasterised OSM prior is what our route head does with SD raster tokens, and the comparison supports it (+5.5 mIoU over CNN concat).

## Needs

rasterised SD map in the ego frame.

## Conflicts

weak alignment is tolerated, not corrected.

## Tried in our experiments

E40 / E48 / E104 (same family; cross-attention kept).
