# C14 - PLUTO: reference lines from the route lane graph as lateral queries

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: PLUTO: Pushing the Limit of Imitation Learning-based Planning (Cheng et al., 2024) https://arxiv.org/abs/2404.14327
- depth: M
- stage: S10, L
- idea: Lanes within a radius are chained by depth-first search along lane connectivity into reference lines (truncated to a uniform length, resampled); embedded reference lines become lateral queries, combined with learnable longitudinal queries into a grid, factorised self-attention along each axis, cross-attention to scene. A separate head decodes one trajectory without a reference line. The imitation loss projects the GT end point to the nearest reference line and selects the matching longitudinal query; contrastive learning uses positive augmentations (state perturbation, non-interactive agent dropout) and negative ones (leading agent dropout / insertion, traffic-light inversion).
- results: nuPlan Val14 score 93.21 (R-score 92.06) vs PDM-Closed 93.08, PlanTF-H 89.96, expert 93.68; without post-processing 89.04. Runtime not reported in the paper content (a secondary table elsewhere lists 250 ms - not verified). Wrong route not addressed.
- differentiable: yes for the network; post-processing is a rule.
- fit_to_us: reuse_idea - effort M - route as chains of lane centre-lines used as queries is the lane-attribute interface; our graph already gives chains; a "no reference line" fallback head is the abstention path.
- tried_before: none. verified: partly
