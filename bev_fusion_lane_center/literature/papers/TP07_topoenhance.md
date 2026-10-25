# TP07 - TopoEnhance (post-hoc graph refinement of link scores, noise-simulating training)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (appendix read from offset 100000: loss, jitter and backbone ablations seen; definitions of TJS / Upper-Bound and the false-positive / curriculum ablations not found in the digest). Pipeline stage: S6 L E.*

## Papers

Refine Connections, Close the Gap: A Reliable Enhancement Framework for Driving Scene Topology (Wang et al., Bosch / MIT, arXiv 2610.11058, 12 Oct 2026) https://arxiv.org/abs/2610.11058 (the digest calls the method "TopoEnhance")

## Idea

Raw pair scores of lane topology methods are badly calibrated, so the thresholded edge set (tau = 0.5) is poor even when TOP (a ranking metric) looks good. A small heterogeneous graph-attention network is trained to re-score ALL pairs of detected nodes from node geometry (and traffic-element embeddings), on training data in which ground-truth nodes are corrupted by simulated false positives and jitter (adaptive curriculum). It is applied after any trained detector without retraining it.

## Formulation

nodes: lanes (3D polylines) and traffic elements (2D boxes + DINOv2 embedding), candidate edges = all pairs (the detector's adjacency is not used). Forward process: false positive v_fp = v + eps with dist > delta, eps ~ N(0, sigma_fp^2 I); jitter v' = v + eps with dist <= delta; edges touching a false positive get label 0; curriculum: sigma_tp rises from 1e-5 toward delta, sigma_fp falls from 10 toward delta. Reverse: two-layer GATv2 (hidden 64, 8 heads, dropout 0.1) with relation set {ll, lt}; refined score s~ = sigmoid(cos(h_u, h_v)). Fusion: s_final = w_r s~ + (1 - w_r) s_raw (w 0.6 for lane-traffic, 0.6 .. 0.9 for lane-lane). Loss: BCE per relation, normalised by the number of candidate edges of that relation (so the larger lane-lane set does not dominate); no Hungarian in training. New metric TJS = |E* intersect E^| / |E* union E^| on thresholded edges; Upper-Bound = oracle edges between true-positive detections.

## Input

detected lanes (polylines), traffic elements, raw pair scores.

## Output

calibrated pair scores and thresholded edge set.

## Training

noise-simulation on GT graphs, AdamW 1e-3, batch 64, cosine to 1e-4, one H200 about 16 h; one model per dataset.

## Differentiable?

yes (small GNN) - the threshold at inference is a rule.

## Sensors and data

any detector output; OpenLane-V2 A and B (v2.1 metrics).

## Reported results

TOP_ll before -> after (upper bound), subset A: TopoNet 10.9 -> 21.8 (24.4), TopoMLP 21.6 -> 24.4 (25.0), TopoLogic 23.9 -> 24.5 (26.0), SMART 37.0 -> 40.1 (40.9). TJS_ll (A): TopoNet 16.3 -> 32.9 (44.3), TopoMLP 18.5 -> 59.7 (69.7), TopoLogic 21.0 -> 43.9 (50.9), SMART 38.1 -> 87.7 (92.9). Subset B: TopoNet TOP_ll 6.7 -> 19.5, TJS_ll 9.0 -> 42.3. Appendix ablations (subset A, TopoNet host, TOP_ll / TOP_lt): plain BCE 12.1 / 22.0, BCE + focal 22.2 / 22.0, per-relation normalised BCE 21.8 / 25.8 (so the normalisation, not the focal term, is what matters); without geometric jitter 21.7 / 25.4; traffic-element embedding backbone irrelevant (DINOv2-S 20.3 .. ViT-L 21.84, ResNet-50 21.8). Ablations of the false-positive synthesis, curriculum, fusion weight and GAT depth were not found in the part read.

## Compute

GAT about 0.1 ms per sample in the main text; stress test 200 lanes + 100 traffic elements = about 60,000 candidate pairs: 6.43 ms on one H200 (message passing 5.2 ms), +0.3 MB memory (the appendix breakdown does not sum exactly).

## Code

none found for TopoEnhance (only TopoPoint's repo is linked as a baseline)

## Fit to our project

reuse_idea - effort M - train the junction / piece verifier (E113 / E116) and link head on SIMULATED corruptions of ground-truth vertex sets (false vertices, jitter, curriculum) with per-relation normalised BCE; add TJS-like edge-set Jaccard to our metrics.

## Needs

a set of nodes with geometry; pairwise candidate edges; a threshold.

## Conflicts

refines only pairs, not geometry; evaluated on query-decoder detections with many false positives (our learned graph has 1.8 x too many junctions, similar regime).

## Tried in our experiments

E113 / E116 (junction / piece verifier on the learned graph): partial (split / merge 0.385 / 0.428 -> 0.437 / 0.504).
