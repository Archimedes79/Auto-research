# SN35 - FIN: camera-radar map segmentation with a distance analysis

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

FIN: Fast Inference Network for Map Segmentation (arXiv 2025) https://arxiv.org/abs/2510.00651

## Idea

CRN-style radar-assisted view transformation and multi-modal deformable cross-attention with a ResNet-50 image backbone and a PAN radar backbone, read by a small two-level U-Net head; the paper evaluates map IoU in 10 m distance steps from 10 to 50 m.

## Formulation

images 256 x 704; BEV 200 x 200 at 0.5 m (the text also says "200 m x 200 m grid", inconsistent with its -50..50 m figures); six classes (drivable, ped crossing, sidewalk, stop line, carpark, divider); six-term loss with a normalised boundary loss (Table 4: focal alone 46.3 mIoU, full set 52.4, + boundary 53.5; the text cites 49.6 for focal alone); head ablation: plain conv head 52.3 mIoU (1.2 M parameters) vs U-Net head 53.5 (1.7 M).

## Input

6 images + 5 radars.

## Output

6 map layers at 200 x 200.

## Training

regime matching_free; nuScenes, 24 epochs in the ablation.

## Differentiable?

yes.

## Sensors and data

camera + radar, nuScenes train / val, rainy and night subsets built from metadata.

## Reported results

val mIoU 53.5; per class drivable 67.7, ped crossing 53.1, sidewalk 60.7, stop line 45.6, carpark 51.3, divider 48.2; 38 ms (A100), 58.4 M parameters. Per-distance numbers exist only in figures: degradation with distance is "proportional across classes", 0 - 25 m named the safe range, night worst; authors list occluded and distant structure as unsolved.

## Compute

58.4 M parameters, 26.3 FPS (A100).

## Code

not checked.

## Fit to our project

reuse_idea - effort S - a U-Net head on a camera-radar BEV is the same read-out as our dense head; no per-range numbers are given for lanes anywhere in the camera-radar literature.

## Needs

images (RVT).

## Conflicts

none new.

## Tried in our experiments

E17 / E18 (U-Net head kept).
