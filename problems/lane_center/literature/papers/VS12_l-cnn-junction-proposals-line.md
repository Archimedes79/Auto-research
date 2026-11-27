# VS12 - L-CNN: junction proposals + line sampler + LoI verification

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, S4, L.*

## Papers

End-to-end wireframe parsing (Zhou, Qi, Ma, ICCV 2019) https://arxiv.org/abs/1905.03246

## Idea

Junctions are detected on a 128 x 128 bin grid; every pair of the top-300 junctions is a line hypothesis; each hypothesis gets ONE score from a verification network that sees 32 bilinearly sampled feature vectors along the segment (line-of-interest pooling).

## Formulation

LoI: N_p = 32 points q_k between the endpoints (linear interpolation), bilinear read of the backbone feature (256 ch reduced by a 1x1 conv to 128), 1-D max-pool stride 4 (32 -> 8 points), flatten 128 x 8, FC(1024) + ReLU, FC -> logit; the feature-map size used for pooling is not stated (junction grid 128 x 128, input 512 x 512). Loss: sigmoid BCE, mean over positives plus mean over negatives per image (balance); multi-task weights junction 8, offset 0.25, verification 1. Junction head: bin likelihood BCE + offset L2 in [-1/2, 1/2); 3x3 max-pool NMS; top K = 300. Sampler: static (300 positives from GT lines, 40 hard negatives from 2,000 pairs ranked by density of the rasterised GT) and dynamic from predicted junctions (match threshold eta not stated): D+ 300, D- 80, D* 600 per image. Inference: all pairs of 300 junctions. Adam 4e-4, wd 1e-4, batch 6, 16 epochs (about 8 h on one GTX 1080 Ti), stacked hourglass.

## Input

bev_raster-like feature map + point_set of junction pairs. output: segments with scores.

## Training

BCE on thresholded junction matches; matching_free.

## Differentiable?

partial (junction NMS / top-K and the sampler are discrete; the verifier is differentiable in the features).

## Sensors and data

images (ShanghaiTech wireframe, York Urban).

## Reported results

ShanghaiTech sAP10 62.9, APH 82.8, FH 81.2; York Urban sAP10 26.4, APH 59.8 (Table 2); post-processing lifts FH 76.9 -> 81.2. Ablation facts seen: N_p was NOT ablated; a 1-D conv instead of the FC head lowers sAP5 57.4 -> 55.7; hand-made coordinate / slope features do not help; hard-negative samplers help. FPS not stated (HAWP reports 15.6 FPS for a re-trained L-CNN).

## Compute

not stated.

## Code

URL from memory, not opened: https://github.com/zhou13/lcnn

## Fit to our project

baseline - S - single-score verifier on the same teardrop ray (the earlier review demanded this comparison); pair-then-verify is also the published hypothesis-selection precedent.

## Needs

both endpoints known before verification (junctions); fixed 32 samples whatever the length.

## Conflicts

one score per line cannot say WHERE a long line stops being a lane; our ray's endpoint is unknown.

## Tried in our experiments

E116 junction / piece verifier on the learned graph (verifier split / merge 0.385 / 0.428 -> 0.437 / 0.504, partial); per-ray survival not tried.
