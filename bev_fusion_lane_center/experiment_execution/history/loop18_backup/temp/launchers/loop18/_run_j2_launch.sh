#!/bin/bash
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
L=$R/temp/launchers/loop18
(bash $L/_run_j2.sh a 0; bash $L/_run_j2.sh a 1) &
(bash $L/_run_j2.sh c 0; bash $L/_run_j2.sh c 1) &
python3 $R/tools/experiment/run.py --id E214.eval --gpu-max-mb 9500 --retries 2 --basis E214.a0,E214.c0,E214.a1,E214.c1 --seed 0 \
  --hyp "evaluation of the joint fine-tune arms against the base and the control on the failure series" -- bash $L/_eval_j2.sh > $R/runs/e17/logs/e214_eval.log 2>&1 &
wait
