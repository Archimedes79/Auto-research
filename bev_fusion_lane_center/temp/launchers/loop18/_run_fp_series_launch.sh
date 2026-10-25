#!/bin/bash
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
python3 $R/tools/experiment/run.py --id E218.a --gpu-max-mb 9500 --retries 2 --basis E215,E213.b --seed 0 \
  --hyp "fork pruning holds under sensor failure and in the best chain" -- bash $R/temp/launchers/loop18/_run_fp_series.sh > $R/runs/e17/logs/e218_a.log 2>&1
