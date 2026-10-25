#!/bin/bash
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
python3 $R/tools/experiment/run.py --id E213.a --gpu-max-mb 9500 --retries 2 --basis E207,E206,E210 --seed 0 \
  --hyp "rough localisation (K145) + trust gate (K146) restore the map benefit at 1-3 m map error and switch off wrong maps; builder B unchanged" -- \
  bash $R/temp/launchers/loop18/_run_r21.sh > $R/runs/e17/logs/e213_a.log 2>&1
