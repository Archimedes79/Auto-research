# GR07 - Trexplorer / Trexplorer Super (recurrent DETR tree tracker, chained sub-trajectory training)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M (Super) / A (base paper: MICCAI page with abstract, reviews and rebuttal)** (A abstract, M method, C code page). Verified: partly (Super paper read; base paper known from the Super text and listings).. Pipeline stage: S4, S5, T.*

## Papers

Trexplorer: Recurrent DETR for Topologically Correct Tree Centerline Tracking (Naeem et al., MICCAI 2024; no arXiv found) https://research.chalmers.se/en/publication/544069 ; Trexplorer Super: Topologically Correct Centerline Tree Tracking of Tubular Objects in CT Volumes (MICCAI 2025) https://arxiv.org/abs/2507.10881

## Idea

Medical analogue of RNGDet in 3D: from a root point a DETR decoder tracks a vessel / airway tree level by level (breadth-first); every patch emits the next nodes of all active branches, branches stop at predicted end nodes and split at bifurcations. The base model adapts the TrackFormer idea (queries carried from step to step) and gives each bifurcation a *fixed number of extra queries* (nb = 2 on the synthetic set per the rebuttal; should be at least the maximum bifurcation degree) that attend to each other and are penalised by the Hungarian loss for following the same branch. Super adds chained training over several sub-trajectories, so that the query outputs of one patch feed the next patch during training, and target augmentation; this removes duplicate branches and premature stops.

## Formulation

base model (as described in the Super paper): nodes [x, y, z, r], about one voxel from the parent, labelled end / intermediate / bifurcation; each patch tracks 9 steps; new patches start at endpoints with a past-trajectory token (up to 10 previous positions); at a bifurcation the branch halts and 26 object queries are assigned (some track the new branches, the rest are discarded); up to 196 concurrent tokens; Hungarian loss + self-attention among queries against duplicates. Super: Super Trajectory Training = 54-node trajectories cut into six overlapping 10-node sub-trajectories with linked nodes, query outputs passed along the chain; Focal Cross Attention = large high-resolution region extracted, cross-attention restricted to a small focal window; Target Augmentation = bifurcation offsets drawn from a Laplace distribution scaled with the radius + Gaussian noise (sigma 0.025) on other points. Inference reuses the query outputs patch to patch from the root.

## Input

3D CT patches + past-trajectory token; `image_feat` (volumes).

## Output

tree graph with radius; per patch the next nodes of all branches.

## Training

`hungarian_set` with chained multi-patch training (`free_rollout`-like unrolling through query states; whether gradients pass between sub-trajectories is not stated); about 2 M iterations on 4 A100.

## Differentiable?

partial - the query state chain is trainable across sub-trajectories (stated as fed forward); Hungarian matching, end / bifurcation thresholds and patch placement are discrete.

## Sensors and data

CT; synthetic vessels (368 / 32 / 100), ATM'22 airways (220 / 16 / 60), Parse 2022 arteries (72 / 8 / 20), 0.5 mm.

## Reported results

point-F1 (1.5 voxel, one-to-one) Trexplorer Super 77.83 / 60.45 / 39.46 vs Trexplorer 39.40 / 3.34 / 10.01 on synthetic / ATM'22 / Parse; branch-F1 77.12 / 41.15 / 23.46; Betti-0 and Betti-1 errors 0 for both Trexplorer models (Tables 2 - 3). Ablation: chained trajectories most important (row mapping ambiguous).

## Compute

not stated.

## Code

https://github.com/RomStriker/Trexplorer-Super ; base model https://github.com/RomStriker/Trexplorer

## Fit to our project

reuse_idea - effort M. Evidence that (a) chained multi-step training of a level-synchronous tree tracker cures duplicates and early stops, (b) reserving a fixed block of queries for a bifurcation (b slots per tip that must separate by loss and self-attention) is workable - the closest published analogue of a fixed-shape b-ary expansion, (c) a metric without linear assignment (1.5 voxel one-to-one) exists.

## Needs

recurrent query state; patch-wise features (our cached pyramid replaces the patch); Hungarian loss.

## Conflicts

Hungarian matching; 3D medical domain; whether gradient crosses sub-trajectories unclear.

## Tried in our experiments

none.
