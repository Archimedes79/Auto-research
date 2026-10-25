# VM44 - mAS (Stability Under Scrutiny)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (page truncated at 100 k of 134 k characters).. Pipeline stage: E.*

## Papers

Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping (Shan et al., ICLR 2026, arXiv 2510.10660) https://arxiv.org/abs/2510.10660

## Idea

A frame-pair stability benchmark. For an anchor frame t one frame t+k (k random in 1..M, M = 2) is drawn; predictions are matched to GT instances by Hungarian in each frame; predictions that map to the same GT instance in both frames are paired; the earlier one is moved by the ego pose into the later frame; presence, localisation and shape scores are combined into a per-instance stability.

## Formulation

presence = 1 if both frames agree on the detection decision at threshold tau, else 0.5 (tau value not given); localisation: both polylines clipped to the range and sampled at N = 100 equidistant x values, Loc = beta * mean |y_{t+k}(x_i) - y_t(x_i)| with beta = 15 (the printed formula lacks the exponential the text mentions); shape = 1 - |kappa_{t+k} - kappa_t| / pi with kappa the mean turning angle; per instance Stability = Presence * [0.7 Loc + 0.3 Shape]; mAS = mean over classes of the class means (not an AP-style PR curve).

## Input

per-frame predictions, GT instances, ego poses.

## Output

scores (presence, localisation, shape, mAS).

## Training

none.

## Differentiable?

no.

## Sensors and data

nuScenes val (no geo split mentioned); uses GT for pairing.

## Reported results

(nuScenes val, M = 2) MapTR (camera, GKT) mAP 44.1 / mAS 71.6; MapTR (C&L) 62.8 / 74.0; StreamMapNet 63.3 / 91.9; MapTracker (R50) 75.95 / 90.4; range of mAS 66.6 - 91.9 in one section, 71.6 - 91.9 in another. mAP and mAS are only loosely related; BeMapNet and PivotNet get 100 % presence by construction (dynamic vectorised sequence); temporal fusion helps MapTR with BEVFormer (71.3 -> 73.0) and hurts with GKT (71.6 -> 66.6); R18 -> R50 lowers MapTR localisation stability (75.0 -> 65.4); 42 models and variants over 10 families.

## Compute

not applicable.

## Code

toolkit announced at https://stablehdmap.github.io/ (release not checked)

## Fit to our project

reuse_idea - M - adapt as a temporal metric on our output: for each frame pair use ego motion, pair elements through GT, and compare presence (validity agreement), position offset along the polyline and turning-angle difference. Our "temporal consistency" metric in PROJECT_CONTEXT has no definition there; this gives one.

## Needs

per-frame predictions with identity via GT; ego pose.

## Conflicts

designed for ordered 3-class polylines (the x-sampling assumes a line that is a function of x within the range).

## Tried in our experiments

temporal consistency is listed among our metrics; its definition is not in PROJECT_CONTEXT.
