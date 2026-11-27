# MH11 - MultiPath++ (learned anchors, hard assignment by distance, aggregation of many modes to few)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, S4.*

## Papers

MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction (Varadarajan et al., ICRA 2022) https://arxiv.org/abs/2111.14973 (arXiv HTML; last 3.5 k characters not read)

## Idea

Replace fixed anchors by learned, input-independent anchor embeddings (one per mixture component); train with the hard assignment of the GT to the component with the smallest Euclidean distance; at the end reduce a large union of modes (several heads) to the wanted few by a likelihood-weighted greedy selection plus EM-style updates.

## Formulation

M embeddings e_1..e_M conditioned on the scene by multi-context-gating blocks; an MLP outputs means, stds, correlation and log-weights per mode (weights tied over time). Loss: maximise the likelihood of the GT under the hard-assigned component (smallest Euclidean distance); Adam, lr 2e-4, batch 64, decay 0.5 every 2 epochs, 800 k steps. Aggregation: bag E heads of L modes -> M' = L * E modes with likelihood / E; greedy centroids (each maximises the likelihood mass within tau of the selected set; submodular, >= 1 - 1/e of optimum), then EM-style updates with hard responsibilities; trained end to end as a final layer; M = 6. - input / output: encoded agent / road / history contexts -> M trajectory modes with Gaussians and weights.

## Training

regime `wta` (hard assignment), no matching between modes and GT.

## Differentiable?

partial - assignment constant; aggregation is a trainable final layer per the paper.

## Sensors and data

Waymo Open Motion, Argoverse.

## Reported results

Waymo validation ablation (k = 6, 8 s): learned anchors minDE 2.305 / MR 0.44 vs static k-means anchors 2.99 / 0.578; test leaderboard (Nov 2021) mAP 0.409, minADE 0.556, MR 0.134. The paper reports that learned anchors now beat static ones and lists richer inputs, architecture and larger batches as possible reasons (study left open).

## Compute

not stated.

## Code

not checked

## Fit to our project

reuse_idea - effort S - per-slot child queries with hard nearest assignment is the cheapest WTA implementation; aggregation corresponds to our "redundancy suppression".

## Needs

per-slot queries conditioned on the tip context; Euclidean distance to the GT.

## Conflicts

aggregation is likelihood-based and run at inference; no diversity beyond WTA.

## Tried in our experiments

none
