# VS34 - UFLD: row-anchor classification with an explicit "no lane" cell

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S3.*

## Papers

Ultra fast structure-aware deep lane detection (Qin, Wang, Li, ECCV 2020) https://arxiv.org/abs/2004.11757

## Idea

Per lane and per predefined row the network classifies WHICH grid cell the lane crosses, with one extra class for "no lane in this row"; structural losses regularise neighbouring rows.

## Formulation

output C x h x (w + 1); L_cls = sum CE over cell index; L_sim = sum_j |P_{i,j,:} - P_{i,j+1,:}|_1; L_shp = sum_j |(Loc_j - Loc_{j+1}) - (Loc_{j+1} - Loc_{j+2})|_1 with Loc = softmax expectation over the w cells (background cell excluded; differentiable, sub-cell); total L_cls + alpha L_str + beta L_seg, alpha = beta = 1, lambda = 1; auxiliary segmentation branch in training only; TuSimple rows 160..710 step 10, 100 cells; CULane rows 260..530 step 10, 150 cells; input 288 x 800. Output count 1.7e4 vs 1.15e6 for segmentation (CULane setting).

## Input

global image_feat. output: per row cell index or none.

## Training

CE; no matching.

## Differentiable?

yes (softmax expectation).

## Sensors and data

images (TuSimple, CULane).

## Reported results

TuSimple accuracy 96.06 % (R34, 5.9 ms), 95.87 % (R18, 3.2 ms); CULane F1 72.3 (R34, 175.4 FPS), 68.4 (R18, 322.5 FPS).

## Compute

ms above.

## Code

URL from memory, not opened: https://github.com/cfzd/Ultra-Fast-Lane-Detection

## Fit to our project

reuse_idea - S - per-row "no lane" = per-sample termination class; the second-order smoothness loss on expected positions is a cheap matching-free regulariser for sampled polylines.

## Needs

a monotone row axis.

## Conflicts

BEV lanes of arbitrary heading, forks, loops.

## Tried in our experiments

none.
