# C31 - LightGlue: matchability scores instead of Sinkhorn, early exit and pruning

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: LightGlue: Local Feature Matching at Light Speed (Lindenberger et al., ICCV 2023, older anchor) https://arxiv.org/abs/2306.13643 ; code https://github.com/cvg/LightGlue
- depth: M
- stage: S6, S9
- idea: Per-point matchability sigma_i = sigmoid(linear(x_i)); assignment = sigma_i sigma_j * softmax_row * softmax_col ("double softmax", no Sinkhorn); a match needs score above tau = 0.1 and mutual best; per-layer confidence head (not back-propagated into the states) allows early exit once 95 % of the points are confident; confident low-matchability points are pruned (beta = 0.01); 9 layers, 4 heads, d = 256.
- results: relative pose MegaDepth-1500 44.2 - 44.5 ms vs SuperGlue 70.0 ms, adaptive 31.4 ms; adaptive depth / width about 33 % faster overall.
- differentiable: yes (thresholding only at inference).
- fit_to_us: reuse_idea - effort S - a sigmoid "has a road on the map" head per vertex (matchability) plus a double softmax is the cheapest abstain-capable assignment; the per-layer exit confidence is not worth it at our size.
- tried_before: none (validity / keep gate E120 exists for vertices, not for the map assignment). verified: partly
