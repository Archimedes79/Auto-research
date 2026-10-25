# GR30 - PAMR (segment-wise autoregressive mapping with a persistent map cache; vision-language decoder)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (paper HTML; table columns garbled, so the cache number is flagged).. Pipeline stage: S8, X.*

## Papers

PAMR: Persistent Autoregressive Mapping with Traffic Rules for Autonomous Driving (arXiv 2509.22756) https://arxiv.org/abs/2509.22756

## Idea

A vision-language model (Qwen2-VL-2B backbone) reads up to 10 front-view images plus quantised pose tokens and writes lane vectors and traffic rules as one token sequence for a 22.4 m x 22.4 m map segment. Segments are chained: after each segment the vector points near its border are projected into the next segment and used as its initialisation (map-rule cache, 10 % overlap; the succeeding prediction wins in the overlap).

## Formulation

tokens: `[lane] ... [/lane]` with quantised coordinates, `[rule]` templates; deterministic parser; prompts (coordinate, box or ROI features) select the target sign. Loss, teacher forcing, decoding: not stated. Training: batch 256, AdamW, lr 2e-5, 20 epochs, 32 H20 GPUs. Ablation on the MapDRv2 test set: map-rule cache off / on 41.24 / 71.32 (HMA; the on-value equals the full model's HMA precision, so the printed column may be precision); segment width 122 / 224 / 448 px: 53.60 / 70.37 / 59.64; overlap 5 / 10 / 15 %: 58.20 / 70.37 / 51.82; 5 / 10 / 15 frames: 64.66 / 70.37 / 64.68.

## Input

perspective images (644 x 364), not BEV; `image_feat` + pose tokens. output: `token_sequence`.

## Training

`next_token`; no matching. differentiable: no (tokens).

## Sensors and data

front camera sequences, MapDRv2 (re-annotated lane geometry plus rules); topology (forks, merges) not described.

## Reported results

HMA F1 70.37 vs 65.29 / 64.20 for two MapDR baselines; rule-extraction F1 83.39 (single-task baseline 89.30).

## Code

project page https://miv-xjtu.github.io/PAMR/ (release not verified)

## Fit to our project

reuse_idea - effort S: border-point hand-over between consecutive windows and its large measured effect (cache on / off) is the only AR-growth data point on *temporal / spatial carry-over*; the model class is not suitable.

## Needs

ego poses; window overlap. compat_conflicts: VLM, perspective input, token decoding.

## Tried in our experiments

temporal carry-over by ego-warped previous set is in the spec (S8); E41 temporal feature fusion +0.008 F1.
