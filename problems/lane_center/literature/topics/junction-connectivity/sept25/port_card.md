# sept25 port card

Verdict: watch. One cheap block worth a test (junction keypoint auxiliary), one conditional block (map-branch gate), one idea that is NOT in the paper but is what the junction problem needs (marked "ours").

## Block 1: junction keypoint auxiliary (IKPD-like)

Interface (our code, names are proposals):
```
JunctionKeypointHead(c_in) : feat [B, C, H, W] -> logits [B, 1, H, W]     # few residual depthwise-separable blocks + 1x1
junction_targets(nodes_xy [N, 2] ego metres, H, W, cell_m, sigma_m) -> [1, H, W] in 0..1, peak 1.0 at the node cell, max over nodes
heatmap_focal(logits, target, alpha=2, beta=4) -> scalar      # CenterNet-style variant is my assumption; the paper says only "focal loss"
```
Where it hooks: the grid feature that feeds the lane head, as a side branch (gradient into the shared feature). Targets: nodes of the map graph where degree is not 2 (merge, diverge, cross), transformed by the registration; or the junction nodes of the GT lane graph (no map dependence). The paper takes the SD map's points, not the HD points.

Minimal spec test (toy, known answer, no data):
1. Graph test: a chain of 5 nodes gives 0 keypoints; a Y (3 links at one node) gives 1; a cross (4 links) gives 1; a diverge and a merge in one chain give 2.
2. Target test: 8 x 8 grid, cell 1 m, one node at the centre of cell (3, 4): target[3, 4] = 1.0 exactly, the neighbours equal exp(-d^2 / (2 sigma^2)) for d = 1 m, target sums to the numpy reference.
3. Loss test: loss is minimal and finite when logits = logit(clipped target), larger for a shifted heatmap; the gradient on the input feature is non-zero (rule: a loss reaches every module).
The paper's own numbers cannot be reproduced without OpenLane-V2, a host and the missing code; the effect to look for is the paper's +1.2 TOP_lsls (Table II, one host, one run).

Cost: a few conv layers on the grid; time not reported by the paper, measure in ms per frame. Risk: our lane head already has node supervision (local digest TP01, section 7: key-point supervision lifted split / merge F1 at 6 m from 0.34 / 0.38 to 0.55 / 0.61), so the auxiliary may only duplicate it; targets from the SD map disagree with the road when the map is outdated (the paper's own failure row), so prefer GT-lane-graph targets for the first test. Effort S.

## Block 2: gated fusion of two map branches (DGFF-like), conditional

Only if both a raster map branch (E104b-style tokens) and a vector branch (E273 tokens) feed the same feature. Interface: `dgff(f_r [B,C,H,W], f_v [B,C,H,W], mu=0.5, nu=0.5) -> [B,C,H,W]` as Eq. 2-4 (concat, FFN, sigmoid gates from the unfused features, two projections, weighted sum). Spec test, numpy reference: with identity projections and gates forced to 1, output = mu * fused + nu * fused = fused for mu + nu = 1; with a zero vector branch and gate_V = 0 the output equals the raster path only. Paper evidence (Table IV, LaneSegNet host): TOP_lsls 31.0 for DGFF against 30.0 for cross-attention and 29.9 for concatenation, one run each. Effort S to M. Leave it unless Block 1 or the map experiments show two branches are in play.

## Idea "ours" (not SEPT): SD-graph-consistent pair prior at junctions

SEPT never uses the connection between roads to constrain which entry joins which exit; the host heads decide pairs. Our map already carries successor / predecessor edges (E273 section 2.5, with its structure term: nodes joined by a link must land on map pieces joined by map edges). The step to test for the junction gap (oracle 0.88 / all pairs 0.53 / learned 0.71 as given in the brief): add the map adjacency C as a log-prior to the entry-exit pair logits, logit_ij += log(eps + C[piece(i), piece(j)]), with the assignment (registration and trust) providing piece(.). Spec test (toy): one 3-way junction, entries A, B, exits C, D; map allows A to C and B to D only; uniform logits with all four pairs above threshold give P 0.5, R 1.0, F1 0.667, and with the prior (eps small) the pairs A-D and B-C fall below threshold, F1 1.0; with a map error (A mapped to B's piece) the F1 drops, showing the trust dependency. This is a design suggestion from my reading, not a result of the paper; it belongs to a different note or card.

## Take / leave

Take: the keypoint auxiliary as a test (GT-graph targets), the finding that vector map tokens beat raster for topology, the idea of a learned gate between map branches. Leave: the OpenLane-V2 heads and their pair scorers, the FiLM raster branch (we register the map by learning), OSM data, DGFF without two branches. What the note cannot supply: any pair mechanism.
