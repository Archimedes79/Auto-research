# GR20 - VecRoad (point-based iterative exploration with flexible step size)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **A+C** (A abstract, M method, C code page). Verified: partly (project page, README, abstract-level text).. Pipeline stage: S4.*

## Papers

VecRoad: Point-Based Iterative Graph Exploration for Road Graphs Extraction (Tan et al., CVPR 2020) https://mmcheng.net/vecroad/

## Idea

Each next move is one *point* that encodes direction and step length together (variable step), guided by jointly predicted road-segmentation and junction cues so that the walk stays aligned; segmentation network and exploration are trained together with summed equal-weight losses (author comment on the project page).

## Formulation

not read (method section PDF-only). Second-hand from the RNGDet text: next-step vertices are extracted as local peaks of a predicted heat map, step size is flexible, the pipeline is two-stage (not end to end), and it may fail for vertices that lie very close together; APLS 65.69 in RNGDet's Table I (RoadTracer set). - input/output/training/differentiable: not read; training: segmentation network and exploration trained together with summed, equally weighted losses (author comment on the project page); dynamic-label or DAgger use not confirmed.

## Sensors and data

aerial, RoadTracer-set (pixel F and junction F compared against RoadTracer).

## Reported results

none seen from the paper itself (APLS 65.69 second-hand, see above).

## Compute

not stated.

## Code

https://github.com/tansor/VecRoad (PyTorch >= 0.4.1, `infer.py`, `train.py`)

## Fit to our project

reuse_idea - effort S - variable step sizes as a *point* is the same as our step class + residual; details unread. - compat_needs / compat_conflicts: unknown.

## Tried in our experiments

step class {2, 4, 6} m in E94.
