# VM07 - Sparse4D v2 (recurrent instance propagation)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, S4.*

## Papers

Sparse4D v2: Recurrent Temporal Fusion with Sparse Model (Lin et al., arXiv 2305.14018) https://arxiv.org/abs/2305.14018

## Idea

3D detection instances (anchor box + velocity, feature vector) are propagated recurrently: the most confident instances of frame t-1 are re-projected (anchor moved by velocity x dt, then by the ego rotation / translation; feature copied unchanged; anchor re-encoded) and concatenated with fresh instances. The current frame samples image features only at the current time.

## Formulation

900 instances per frame in training: 600 temporal + 300 single-frame; anchor A = {x,y,z,w,l,h,sin yaw,cos yaw,vx,vy,vz}; Eq. 2: position + v dt, then R, T; yaw vector and velocity rotated by R. Single-frame layer scores and selects the new instances; multi-frame layers have temporal cross-attention + self-attention. Depth reweighting of v1 dropped once dense depth supervision (L1) exists. 100 epochs, AdamW, no CBGS; lr, batch, history length, loss / matching and gradient treatment of carried instances NOT stated on the page.

## Input

multi-view features (R50 / R101 / V2-99 + FPN) + instance memory.

## Output

3D boxes + class + velocity.

## Training

not stated on the page (from memory: Hungarian one-to-one).

## Differentiable?

partial (top-k).

## Sensors and data

camera; nuScenes val / test.

## Reported results

nuScenes val R50 256 x 704: mAP 0.439, NDS 0.539, 20.3 FPS; R101 512 x 1408: 0.485 / 0.580 at 8.4 FPS; test V2-99: 0.557 / 0.638.

## Compute

20.3 FPS (R50).

## Code

https://github.com/linxuewu/Sparse4D

## Fit to our project

reuse_idea - S - the plain template of "ego-warped previous set": store (geometric anchor, feature) per instance, transform only the geometry, keep the feature, concatenate with fresh instances.

## Needs

geometry separable from a feature vector; ego transform; confidence.

## Conflicts

3D boxes, not polylines.

## Tried in our experiments

none.
