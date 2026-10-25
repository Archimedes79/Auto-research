# GR21 - Image-conditioned graph generation (Graph Generative Transformer, StreetMover distance)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked.. Pipeline stage: S4, E, L.*

## Papers

Image-Conditioned Graph Generation for Road Network Extraction (Belli, Kipf, arXiv 1910.14388, 2019; no venue seen) https://arxiv.org/abs/1910.14388

## Idea

A self-attentive decoder emits the graph node by node in BFS order: each step outputs the node coordinates (tanh) and a soft adjacency vector over the M frontier slots (sigmoid), conditioned by attention on a 900-d image vector. Trained by teacher forcing; evaluated with a StreetMover distance - a Sinkhorn approximation of the Wasserstein distance between point clouds sampled on the graph edges - which needs no matching and no post-processing.

## Formulation

12 blocks, d = 256, 8 heads; loss 0.5 BCE(adjacency) + 0.5 MSE(coordinates); BFS from the top-left node with children ordered clockwise relative to the incoming edge; M = frontier width at the 99th percentile (value not stated); greedy thresholding at 0.5 (beam search not tested).

## Input

64 x 64 grayscale semantic segmentation (900-d encoder code); `map_raster`. output: `graph`.

## Training

`teacher_forcing`; `matching_free` loss, ordered by a fixed convention.

## Differentiable?

no at inference (threshold), yes in the loss.

## Sensors and data

Toulouse OSM graphs (111,034 tiles of about 110 m).

## Reported results

StreetMover 0.0158 (GGT) vs 0.0289 (RNN), 0.0245 (modified GraphRNN), 0.0192 (no context attention); fails on cluttered graphs and with noisy input segmentations (no pre-training).

## Compute

one GTX 1080 Ti, 200 epochs. code: none found

## Fit to our project

reuse_idea - effort S: StreetMover / Sinkhorn-OT between sampled graph points as a matching-free metric or loss; frontier-width idea.

## Needs

fixed node order. compat_conflicts: toy-scale input, no real sensor.

## Tried in our experiments

point-to-polyline both ways (E120) is a cheaper cousin.
