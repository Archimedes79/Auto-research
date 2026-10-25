# GR06 - DeH4R (decoupled hybrid: candidate vertices, adjacent-vertex prediction, expansion from dangling ends)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper HTML twice, README).. Pipeline stage: S1, S4, S7.*

## Papers

DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction (Gong, Ji; accepted to IEEE T-GRS; arXiv 2508.13669, Aug 2025, v2 Mar 2026) https://arxiv.org/abs/2508.13669

## Idea

Splits graph growing into parts that run in parallel on one backbone pass: (1) candidate vertices from keypoint / sampling-point / road masks, (2) for every candidate a decoder predicts up to 10 adjacent vertices (relative coordinates + road / non-road), (3) an initial graph by mutual matching of the predicted neighbours, (4) a graph expander that treats all degree-1 vertices as new seeds, inserts or merges predicted vertices, and repeats for at most 3 rounds.

## Formulation

Backbone SAM2 Hiera-B+ -> FPN (1/4, 1/8, 1/16, 1/32). AVP: for each candidate an l x l (l = 3) grid of sample points per scale, bilinear `grid_sample`, scales summed -> f_roi; 3 transformer decoder layers (hidden 128, 8 heads, FFN 256), N_q = 10 learned queries -> p_i in R^4 (2D offset + 2 class logits); overlapping patches are averaged query-wise (queries keep a fixed order). Decoding: candidates v_n, v_c connect if their predicted neighbours point at each other; cost z = s1 + s2 + w (1 - cos th1) + w (1 - cos th2), w = 10 (taken from Sat2Graph). Expansion: valid vertex absent from the graph is inserted with its edge; if within D_merge = 10 px of an existing vertex it is merged; stop when no degree-1 vertex remains or after 3 rounds (Table V: F1 81.14 at 1 round, 81.26 at 6 - 7). Loss L = BCE(three masks) + CE(class) + 10 L2(coordinates), Hungarian matching of predicted to GT adjacent vertices; validity threshold 0.5 (CityScale) / 0.45 (SpaceNet). Training samples = each GT vertex in a patch plus its direct neighbours, N(0,1) noise on adjacent vertices, random 90-degree rotations; no rollout training, exposure bias not discussed.

## Input

RGB patch 512 x 512 (CityScale) / 256 x 256 (SpaceNet), 1 m / px; `image_feat` pyramid.

## Output

graph; per candidate up to 10 neighbours.

## Training

`teacher_forcing`-like (GT vertices as candidates) + `hungarian_set`; SAM2 backbone fine-tuned at 0.1 x lr.

## Differentiable?

no - NMS for candidates, Hungarian, handcrafted decoding and expansion rules; each module is a plain regression head.

## Sensors and data

aerial RGB; CityScale (144 / 9 / 27 tiles), SpaceNet (2,040 / 127 / 382).

## Reported results

CityScale TOPO-F1 / APLS / IoU 81.21 / 72.38 / 55.79 vs RNGDet++ 78.44 / 67.72 / 45.61 and SAM-Road 76.20 / 66.54 / 33.87 (reproduced, Table I); SpaceNet 86.04 / 73.37 vs SAM-Road 82.72 / 71.10. Modes (CityScale): decoding only 81.02 / 71.72, expansion only 79.18 / 68.70, both 81.21 / 72.38 (Table IV; row mapping inferred). ROI 1x1 / 3x3 / 5x5 / 7x7 / 9x9: F1 81.07 / 81.21 / 80.96 / 81.00 / 80.91 (Table VI) - a single sample point almost suffices with a strong backbone. Time 13.34 min vs RNGDet++ 136.0 min (CityScale).

## Compute

not stated (parameters); 4 GPUs for training.

## Code

https://github.com/7777777FAN/DeH4R

## Fit to our project

reuse_idea - effort M. Parallel expansion of all dangling ends in a few rounds (a level-synchronous wavefront like our ring wavefront), grid-sampled multi-scale ROI features (same operation as our bilinear fan), merge-by-radius with insertion, and the "expansion cannot remove wrong edges" caveat. Backbone and image domain not reusable.

## Needs

candidate set; matching of GT neighbours (Hungarian) - replace by sector slots; a merge radius.

## Conflicts

Hungarian set loss; hand rules for decoding; expansion seeds only from degree-1 vertices (cannot revisit wrong edges).

## Tried in our experiments

E94 - E99 (trace network, similar events) partial; E42 / E45 / E46 / E51 / E56 (learned link heads) within 0.01 of the rules, dropped - DeH4R's initial-graph decoding is a hand rule on predicted neighbours.
