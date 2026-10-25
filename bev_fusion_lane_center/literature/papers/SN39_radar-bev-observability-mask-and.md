# SN39 - Radar BEV observability mask and unknown-region hallucination rate

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (mask, loss and metric passages)** (A abstract, M method, C code page). Verified: partly (mask sections only). Pipeline stage: D, L, E.*

## Papers

On Learning Spatial Structure from Pre-Beamforming Per-Antenna Range-Doppler Radar Measurements (Sebastian, Berthold, Forkel, Pohl, Maehlisch; IEEE RA-L 2026) https://arxiv.org/abs/2604.01921

## Idea

For a radar-to-BEV occupancy network, supervision is applied only where the radar field of view overlaps the lidar-observable area; lidar-unobservable cells inside the radar field are excluded from the loss but used to count how often the network "hallucinates" occupied cells there (unknown-region hallucination rate).

## Formulation

lidar rays cast over azimuth bins of 0.05 deg after a height filter removes ground; the nearest non-ground return is the occluder, with no obstacle the farthest return sets the free extent; cells up to the endpoint are observable, beyond it unobserved; masked focal loss over supervised cells only; AP and occupied-class IoU computed inside the supervised region; BEV 0 - 60 m forward, +-38 m lateral, 0.5 m cells (120 x 152).

## Input

radar range-Doppler data.

## Output

BEV occupancy.

## Training

masked supervised; regime matching_free.

## Differentiable?

yes (the mask is a fixed label-side tensor).

## Sensors and data

radar + lidar (lidar for the mask only), the authors' own campus recordings (16,600 frames; 11,680 train / 4,920 validation), not nuScenes.

## Reported results

no numbers seen.

## Compute

not stated.

## Code

not checked.

## Fit to our project

reuse_idea - effort S - gives the pair our censoring needs: a loss mask on unobserved samples plus a metric that counts positives hallucinated in unobserved cells; for us the lidar part would be replaced by the calibration mask (SN13) plus box shadows.

## Needs

a label-side observability map.

## Conflicts

lidar for the mask (project: no lidar input; label-side use is a project decision).

## Tried in our experiments

none.
