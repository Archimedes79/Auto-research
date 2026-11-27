# VS44 - Lane extent / termination heads across lane detectors (synthesis card)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M (through the cards)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, S7.*

## Papers

see VS26 (CLRNet), VS27 (LaneATT), VS33 (CondLaneNet), VS34 (UFLD), VS35 (GANet), VS36 (Anchor3DLane), VS38 (LATR)

## Idea

Every method seen decides where a lane stops with a regressed length or a per-row / per-point binary signal; none uses a cumulative survival product and none separates "lane ends" from "not observable".

## Formulation

LaneATT and CLRNet: length l (end index = start index + l - 1; smooth-L1); CondLaneNet: per-row BCE "lane passes this row" (weight 1.0) plus RIM continue / stop; UFLD: extra (w + 1)-th class per row; Anchor3DLane: visibility per point (L1); LATR: visibility per point (BCE); GANet: lane ends where keypoints stop (no explicit end). - input / output / training / differentiable: see the cards.

## Sensors and data

images; OpenLane 3-D.

## Reported results

none (synthesis).

## Compute

not stated.

## Code

see the cards.

## Fit to our project

reuse_idea - S - per-row BCE existence is a non-cumulative hazard-like signal; length regression is a single-number endpoint.

## Needs

a monotone axis.

## Conflicts

forks.

## Tried in our experiments

E94 end event.
