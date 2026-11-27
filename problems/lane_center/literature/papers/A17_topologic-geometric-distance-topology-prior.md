# A17 - TopoLogic: geometric-distance topology prior (family reference, no memory)

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: TopoLogic: An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes (Fu et al., NeurIPS 2024) https://arxiv.org/abs/2405.14747
- depth: M
- stage: S6
- idea: Connectivity = combination of (i) a learnable monotone function of the distance between the end point of one lane and the start point of the next, f = exp(-x^alpha / (lambda sigma)) (sigma = std of the distance matrix; alpha init 0.2, lambda init 2) and (ii) a semantic similarity from two 3-layer MLPs on the lane queries; G = lambda1 G_dis + lambda2 G_sim. The geometric part can be added to a trained model without retraining.
- formulation: loss on G_sim only; G_dis trained indirectly via the detection loss through the GNN.
- input: lane queries + end-point geometry; output: lane-lane topology matrix.
- training: standard OpenLane-V2.
- differentiable: yes.
- sensors_data: camera, OpenLane-V2.
- results: subset A TOP_ll 23.9 vs TopoNet 10.9; OLS 44.1 vs 39.8; post-hoc use: TopoNet 10.9 -> 22.3, LaneSegNet TOP_lsls 25.4 -> 29.6. Temporal use: none stated in the digest.
- compute: not stated.
- code: https://github.com/Franpin/TopoLogic (as named by search result; not opened)
- fit_to_us: baseline - effort S - relevant to memory only as a topology scorer that is robust to end-point shifts; our rule system already does this.
- compat_needs: lane end points.
- compat_conflicts: none.
- tried_before: E42, E45, E46, E51, E56 TopoLogic-style scoring: within 0.01 of rules, dropped.
- verified: checked
