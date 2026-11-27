# bev_fusion champions

Champion: **open, the owner's choice** between `t21_ctl` (the t20 network continued 1500 steps) and `t20_base_s0` (the reference baseline of
D48). Both are measured through the research system (warm evaluation, ground truth v13, VAL 1000 frames spread over 82 scenes):
E27 `t21_ctl` board7 0.4631 / 0.4635 (seed 0 / 1), E28 `t20_base_s0` 0.4569 - inside the band 0.02 of each other. The older numbers
(`t21_ctl` 0.45489 / 0.45482, `t20_base_s0` 0.44654) are cold reads on ground truth v8 and not comparable with them. On the owner's
decision: `research register bev_fusion E27 t21 --decision D<n> --role champion` (or E28 t20). Proposed comparison networks: GaussianCaR
and CRN.

Code: `../../problems/bev_fusion/`.
