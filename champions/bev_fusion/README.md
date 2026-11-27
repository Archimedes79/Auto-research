# bev_fusion champions

Champion: **open, the owner's choice** between `t21_ctl` (the t20 network continued 1500 steps) and `t20_base_s0` (the reference baseline of
D48). Measured again with the new evaluation (warm, ground truth v13, VAL 1000 frames spread over 82 scenes): `t21_ctl_s0` board7 0.4631.
The older numbers (`t21_ctl` 0.45489 / 0.45482 for seed 0 / 1, `t20_base_s0` 0.44654) are cold reads on ground truth v8 and not comparable
with it. Proposed comparison networks: GaussianCaR and CRN.

Code: `../../problems/bev_fusion/`.
