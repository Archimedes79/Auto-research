# VM17 - InteractionMap (key-frame hierarchical fusion, geometry-aware classification)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, L.*

## Papers

InteractionMap: Improving Online Vectorized HDMap Construction with Interaction (Wu et al., CVPR 2025) https://arxiv.org/abs/2503.21659

## Idea

(1) Key frames chosen by DISTANCE (stride 5 m, N_pre = 4 submaps), local fusion (warp previous BEV + GRU + residual block) into a "submap", global fusion by streaming (GRU over a global BEV) or stacking (warp and concatenate the N_pre submaps). (2) Explicit point-to-instance relation prior added to the attention logits (point level: log-scaled coordinate differences + edge-direction cosine differences; instance level: sign of score difference x Chamfer distance). (3) Geometry-aware classification: the positive target of the focal-style BCE is a soft score s_geo (p2p: 1 - mean Manhattan distance of normalised points; dir: 0.5 + mean edge cosine / 2; GIoU variant) and the same score enters the matching cost.

## Formulation

s_p2p = 1 - (1/(2 N_p)) sum Manhattan(pred, GT) on normalised points; best = s_p2p + s_dir; matching cost C = s_geo BCE(s_geo, p) - alpha p^gamma BCE(1-p, 1). First 50 % of epochs single-frame. Range +-15 x +-30 m, BEV 200 x 100 (0.3 m), R50, AdamW 6e-4 cosine, batch 32, 8 A800, 24 / 110 epochs. One-to-many assignment not described.

## Input

images -> BEV + submap memory.

## Output

polylines + class (MapTRv2 family).

## Training

Hungarian with geometry-aware cost; regime hungarian_set.

## Differentiable?

partial (Hungarian).

## Sensors and data

camera; nuScenes, AV2 (2 Hz, 10 Hz); standard val sets (no new-split row on the page).

## Reported results

nuScenes val: MapTRv2 61.5 -> InteractionMap-R 71.6 / -C 71.8 (24 ep); 68.7 -> 75.5 / 76.0 (110 ep). AV2 2D 2 Hz 30 ep: 64.3 -> 73.7. 11.0 - 12.1 FPS (A800).

## Compute

11.0 - 12.1 FPS; model size 49.9 - 116.7 MB.

## Code

none on the page.

## Fit to our project

reuse_idea - S - (a) key-frame stride by distance (agrees with MapTracker's strided memory), (b) soft quality target for the validity head (same family as VM34 and VM06).

## Needs

ego warp; GRU on BEV; validity head with a soft target. Cached-BEV: the fusion part yes.

## Conflicts

the relation prior and matching cost live inside a DETR decoder.

## Tried in our experiments

E41 (single-memory GRU).
