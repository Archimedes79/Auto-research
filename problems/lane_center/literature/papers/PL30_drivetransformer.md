# PL30 - DriveTransformer (task-parallel queries including map queries)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10.*

## Papers

DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving (ICLR 2025; authors not read) https://arxiv.org/abs/2503.07656

## Idea

Agent queries (900), map queries (100) and one ego query interact in every block (sensor cross-attention, task self-attention, temporal cross-attention to a FIFO history of ego queries). Each map query is replicated per point with a position encoding in the sensor cross-attention and merged back by a PointNet (points per polyline and classes not given). Six planning mode embeddings; winner-take-all training.

## Formulation

ego query from an MLP over CAN-bus info; ego position encoding updated after each block from the predicted trajectory; losses on task heads at all blocks; no detach described.

## Input

sensor tokens.

## Output

ego plan (6 modes), agent and map predictions.

## Training

end-to-end; planning gradients appear to reach map queries (inferred, not confirmed).

## Differentiable?

yes; matching discrete.

## Sensors and data

nuScenes, Bench2Drive.

## Reported results

nuScenes open loop (Large, VAD metric) without ego status L2 0.40 m avg, collision 0.11 %; with ego status 0.33 / 0.07; Bench2Drive driving score 63.46, success 35.01 %.

## Compute

latency 211.7 ms (Bench2Drive table).

## Code

https://github.com/Thinklab-SJTU/DriveTransformer/

## Fit to our project

reuse_idea - effort M - map-instance queries as tokens with per-point position encoding and a PointNet merge.

## Needs

instance queries.

## Conflicts

Hungarian matching.

## Tried in our experiments

none.
