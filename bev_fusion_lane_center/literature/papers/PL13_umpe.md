# PL13 - UMPE (map prior encoder with confidence-biased attention, used with VAD)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S9, S10.*

## Papers

Unified Map Prior Encoder for Mapping and Planning (arXiv 2605.02762; authors not read) https://arxiv.org/abs/2605.02762

## Idea

One prior encoder for HD vectors, OSM vectors, satellite tiles and rasterised SD maps. Vector priors get SE(2) pre-alignment, multi-frequency sinusoidal point encoding and a 6-layer transformer producing tokens + confidences; BEV queries attend with dual cross-attention carrying a log-confidence bias, then presence-normalised channel gating. Raster priors: ResNet-18 with FiLM and grid_sample micro-alignment.

## Formulation

merge X = LN(Y) + alpha W LN(Z), W zero-initialised, alpha ramps 0 to about 0.6; source dropout p = 0.3.

## Input

priors + BEV query features.

## Output

augmented BEV feature for MapTRv2 / MapQR mapping and for VAD's planning decoder.

## Training

from scratch with a two-stage curriculum and separate learning rates for prior branches; whether planning gradients reach the prior encoder is not confirmed.

## Differentiable?

yes (the confidence bias is additive in the attention logits).

## Sensors and data

nuScenes.

## Reported results

planning with VAD: avg L2 0.72 -> 0.42 m, collision 0.22 -> 0.12 % (open loop only; no boundary metric, no NAVSIM); no planner-input noise test.

## Compute

planning runs 60 epochs on 8 x A800.

## Code

https://github.com/Ethan-Zheng136/UMPE

## Fit to our project

reuse_idea - effort S - the log-confidence additive attention bias is the soft form of our validity mask.

## Needs

per-token confidence.

## Conflicts

only VAD tested for planning; SparseDrive / UniAD / DiffusionDrive appear only as table baselines.

## Tried in our experiments

E40 / E48 (map prior at the grid, learned registration) are the closest.
