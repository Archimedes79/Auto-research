# C16 - GC-PGP: goal / route as soft or hard mask on lane-graph edges (older anchor, 2023)

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: From Prediction to Planning With Goal Conditioned Lane Graph Traversals (Janjos et al., ITSC 2023) https://arxiv.org/abs/2302.07753 ; project page https://mh0797.github.io/gc-pgp/
- depth: M
- stage: S9, S10
- idea: PGP samples traversals on a lane graph; the route (nodes on a path from the vehicle to the goal) modifies edge probabilities: soft mask = learnable additive bonus beta on on-route edges, hard mask = off-route edges get probability zero (inference only, or also in training).
- results (nuPlan only, no nuScenes): open-loop intersection split GC-PGP 0.74 vs PGP 0.56 vs Urban Driver 0.83; closed-loop 0.46 vs 0.37 vs Urban Driver 0.58 vs IDM/MOBIL 0.75; plan instability 2.80 m vs 3.65. Some nuPlan route labels were wrong; the hard mask applied during TRAINING was much less robust to such labelling errors.
- differentiable: soft mask yes; hard mask is a rule. code: project page claims code (not opened).
- fit_to_us: reuse_idea - effort S - route as an additive bias on edge probabilities (soft), hard only at inference; directly applicable to our consumer (E117 reads route probability per edge).
- tried_before: none. verified: partly
