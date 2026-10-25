#!/bin/bash
# refinement of K188: forgetting factor of the accumulation (0.8 / 0.9 / 0.97) and the confidence weight (trust x peak vs trust); pose error at sigma 2 m (night and blur failure half); outputs to runs/e18/regis_eval_lam*
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
for lam in 0.8 0.97; do
  bash $R/tools/env/_py14.sh p18_regis_eval.py --regis regis_a --out regis_eval_lam$lam --tags dark_maxdecay_0.95,blur_maxdecay_0.95 --sigmas 2 --wrong 0 --lam $lam 2>&1 | grep "^\[" | sed "s/^/lam $lam /"
done
bash $R/tools/env/_py14.sh p18_regis_eval.py --regis regis_a --out regis_eval_conft --tags dark_maxdecay_0.95,blur_maxdecay_0.95 --sigmas 2 --wrong 0 --conf t 2>&1 | grep "^\[" | sed "s/^/conf=trust only /"
