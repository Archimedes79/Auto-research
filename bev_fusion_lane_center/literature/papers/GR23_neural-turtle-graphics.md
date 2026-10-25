# GR23 - Neural Turtle Graphics (queue-based autoregressive road layout, closure by proximity)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: . Pipeline stage: S4, S7.*

## Papers

Neural Turtle Graphics for Modeling City Road Layouts (Chu et al., ICCV 2019; arXiv 1910.02055) https://arxiv.org/abs/1910.02055

## Idea

Generation pops an unvisited "active node" from a queue; a bidirectional GRU encodes the incoming paths, a GRU decoder emits the *outgoing nodes* one by one as relative offsets (dx, dy) plus a continue flag; outgoing nodes are sorted counter-clockwise during training so no assignment problem arises; a new node within 5 m of an existing one is not queued but joined to it, which closes cycles. For aerial parsing the CNN likelihood is multiplied with the learned prior and nodes are sampled until the maximum new-node probability falls below 0.05.

## Formulation

offsets discrete at 1 m, clipped to +-100 m (city generation), polar coordinates aligned to the previous edge for parsing; K random incoming paths per active node during training; teacher forcing with CE; Adam 1e-3.

## Differentiable?

no (sampling). training: `teacher_forcing`. tried_before: none. verified: checked.

## Sensors and data

OSM (17 cities), SpaceNet.

## Reported results

SpaceNet APLS 74.97 (NTG-I with DLA + STEAL) vs 71.04; unseen-city 57.89 vs 56.15; FID-based city metrics.

## Code

none found (stated as not open in HDMapGen)

## Fit to our project

reuse_idea - effort S: counter-clockwise ordering of children as a way to make b outputs order-fixed without matching; closure-by-proximity rule (5 m) as the first dedup rule; prior x likelihood product as a rescoring form.
