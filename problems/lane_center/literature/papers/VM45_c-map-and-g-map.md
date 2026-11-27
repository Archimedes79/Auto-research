# VM45 - C-mAP and G-mAP (consistency-aware and global-map metrics)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: E, D.*

## Papers

MapTracker (consistency-aware mAP, VM04) https://arxiv.org/abs/2403.15951 ; HisTrackMap (G-mAP, VM15) https://arxiv.org/abs/2503.07168

## Idea

C-mAP: per-frame Chamfer matching as in the usual mAP, but a match is counted false positive if the GT track was earlier matched to a different predicted ID or an unmatched ancestor invalidates it; baselines without IDs get tracks by IoU of rasterised masks with a look-back window; computed on a "consistent GT" in which elements are merged over time. G-mAP: local GT vectors of N_seq frames are transformed to a global frame and rasterised; per-frame predictions are merged with MapTracker's merge function; polygons by rasterised AP, polylines by Chamfer AP with GT points from farthest-point sampling; average over the three categories.

## Formulation

see idea; thresholds of the Chamfer AP as in the standard protocol; look-back window N and track-extraction details not read.

## Input

per-frame predictions with IDs (or extracted tracks), consistent GT, ego poses.

## Output

C-mAP, G-mAP.

## Training

none.

## Differentiable?

no.

## Sensors and data

nuScenes, AV2.

## Reported results

nuScenes C-mAP MapTracker 69.1 vs StreamMapNet 56.4 (consistent GT); non-overlapping 32.5 vs 22.2; G-mAP nuScenes 50.2 (HisTrackMap) vs 49.4 (MapTracker). MapTCL (VM23) and PredMapNet (VM14) also report C-mAP only.

## Compute

not applicable.

## Code

https://github.com/woodfrog/maptracker (evaluation included, not checked)

## Fit to our project

reuse_idea - M - our pieces carry IDs in the GT piece graph, so an ID-consistency score (does the same lane keep the same output piece over time) is definable; merging multi-frame predictions into one global map to test long-horizon consistency is an option for the route task.

## Needs

element identity in output and GT.

## Conflicts

our output vertices have no persistent identity yet.

## Tried in our experiments

none.
