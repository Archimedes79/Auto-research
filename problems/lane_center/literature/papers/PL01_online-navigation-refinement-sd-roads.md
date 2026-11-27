# PL01 - Online Navigation Refinement (MAT on OMA): SD roads to perceived centre-lines, route attribution after perception

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S9.*

## Papers

Online Navigation Refinement: Achieving Lane-Level Guidance by Associating Standard-Definition and Online Perception Maps (authors not read; repo owner WallelWan; ICLR 2026 per repo README) https://arxiv.org/abs/2507.07487

## Idea

Input is an SD road graph (OSM; each road an ordered sequence of directed vectors with connectivity) and an online-perception (OP) map (lane centre-lines sampled uniformly with topology, plus boundary vectors) produced by MapTR / MapTRv2 / SeqGrowGraph at test time. A transformer assigns each perceived centre-line to exactly one SD road; an SD path (the route) then maps to a path of OP centre-lines by topological sorting. This is route attribution after perception, but hard and road-level, not a route probability per vertex.

## Formulation

tokens = vectors [x1, y1, x2, y2, theta] -> 2-layer MLP. Block = Path-Aware Attention (enumerate valid root-to-leaf paths, reorder tokens by path index, grouped attention over adjacent tokens, restore, average over paths) + Spatial Attention (3-D grid x, y, orientation, Hilbert-curve serialisation, grouped attention) + FFN. Head: Prob_ij = softmax over roads j of (feature of centre-line i . mean-pooled road feature j / sqrt d), not Sinkhorn. Post-processing: start at the most confident centre-line, bidirectional beam search constrained by road connectivity. Channels [96, 192, 384, 768, 1536]; blocks MAT-T [2,2,2,2,2], MAT-L [4,4,4,12,4].

## Input

SD road graph (vectors + connectivity) and OP centre-lines + boundaries in the ego frame; GPS pose is not an input (drift is treated as spatial noise).

## Output

per-centre-line road id (unique; a road may own several centre-lines); the route is the SD path mapped to an OP path by topological sorting.

## Training

loss 1.0 CE + 0.01 CTC on hand-annotated associations; dataset OMA built on nuScenes + OSM: over 30 K scenarios, 480 K road paths, 2.6 M lane vectors; train + val 31,724 (GT OP maps), test 5,573 (MapTRv2-generated OP maps), area-disjoint split; 50 epochs AdamW lr 1e-4, batch 128; regime: supervised.

## Differentiable?

partial - the softmax head is differentiable; path extraction (beam search) is not; trained apart from the OP generator, no gradient into it.

## Sensors and data

nuScenes (Boston, Singapore) + OSM; OP maps come from camera BEV models; no radar.

## Reported results

val NR-F1(50:95) MAT-T 78.2, MAT-L 78.7, best baseline 72.9 (EAM3); test with OP maps from MapTR / MapTRv2 / SeqGrowGraph: MAT-L 41.9 / 45.0 / 54.9 vs EAM3 36.3 / 39.1 / 50.8. Imperfect SD maps (appendix, val / test): global shift 10 / 20 / 50 %: 77.5 / 43.3, 76.0 / 42.9, 74.6 / 41.4; element absence up to 30 %: 75.5 / 43.1. Zero-shot nuScenes-C hard: rain 40.6, fog 42.1, dark 41.6. Ablation val: path-aware attention alone 74.1 vs 61.8 plain backbone.

## Compute

latency 34 ms (T), 70 ms (L) on an A6000; parameters not stated.

## Code

https://github.com/WallelWan/OMA-MAT (dataset OMA on Hugging Face `wanjiaxu/OMA`)

## Fit to our project

reuse_idea - effort M - their SD input is road vectors, ours is SD raster tokens, and the association is hard and road-level.

## Needs

OP lane graph as polylines with topology (our assembled graph); SD roads as vectors with connectivity; ego-frame alignment within GPS-level drift.

## Conflicts

hard unique assignment vs our soft route probability; beam-search post-processing not differentiable; trained on GT OP maps, tested on generated ones; separate from perception (no end-to-end).

## Tried in our experiments

E102-E104 (route attribution inside the learned chain; cross-attention kept, route success 0.853-0.820) is the end-to-end analogue; vector-SD association not tried.
