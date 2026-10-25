# C30 - SuperGlue: Sinkhorn optimal transport with a dustbin

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: SuperGlue: Learning Feature Matching with Graph Neural Networks (Sarlin et al., CVPR 2020, older anchor) https://arxiv.org/abs/1911.11763
- depth: M
- stage: S6, S9 (assignment)
- idea: Score matrix S_ij = <f_i^A, f_j^B>, extended by one learnable scalar z in an extra row and column (dustbin); marginals [1_M, N] and [1_N, M]; Sinkhorn with T = 100 iterations; dustbin dropped afterwards; unmatched points get a probability mass in the dustbin, which is the abstention mechanism.
- loss: NLL of ground-truth matches plus NLL of the dustbin entries of unmatched points; keypoints with ambiguous status are left unlabelled. Attentional GNN of 9 self / cross layers (4 heads, d = 256), about 12 M parameters (appendix); about 69 ms per pair on a GTX 1080; test threshold 0.2.
- differentiable: yes (Sinkhorn is differentiable; T = 100 is heavy).
- fit_to_us: reuse_idea - effort S - add ONE dustbin column ("no SD road / off-map") to the vertex -> road softmax with an NLL for lanes that are not on any map road, and a dustbin row for map roads with no perceived lane; two or three Sinkhorn iterations are enough for a (vertices x roads) matrix of a few hundred entries (my judgement, not tested by the paper).
- compat_needs: unmatched labels (lanes without a road, roads without a lane) - we have them from the map expansion.
- compat_conflicts: our losses avoid Hungarian matching; Sinkhorn is not Hungarian but needs one-to-one marginals, whereas road-to-lane is many-to-one (C01) - use dustbin + softmax, not full OT.
- tried_before: none. verified: partly
