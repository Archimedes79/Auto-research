# C32 - Differentiable weighted Procrustes (Deep Global Registration)

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Deep Global Registration (Choy, Dong, Koltun, CVPR 2020, older anchor) https://arxiv.org/abs/2004.11540
- depth: A (abstract page; how the weights are produced and the gradient proof were not on the page)
- stage: S9 (registration step)
- idea: A network predicts a confidence per correspondence; a closed-form weighted Procrustes solves the rigid transform; gradients flow through the weights (not through coordinates), linear in the number of correspondences; a robust gradient-based refinement follows. A search-engine summary of the paper states the weights-not-coordinates property and the linear cost (from the paper's text as quoted there); not verified on the page I opened.
- differentiable: yes in form. SE(2): not covered - no SE(2) variant seen anywhere; the 2-D closed form (2 x 2 SVD, or the atan2 closed form) would need to be re-derived by us (not claimed by any paper I read).
- fit_to_us: reuse_idea - effort S - vertex <-> map-road-point correspondences with soft weights from the assignment softmax give a per-frame or per-road (dx, dy, dtheta) in closed form; cost negligible.
- compat_conflicts: assumes point-to-point correspondences; lanes are curves (point-to-polyline residual, as in our E120, is the better residual).
- tried_before: none. verified: partly (A)
