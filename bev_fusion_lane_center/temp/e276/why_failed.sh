#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
date +%H:%M
for r in E279.w_s0 E279.w_s1; do echo "== $r"; grep -h '"status"\|"exit"\|"reason"\|"note"' experiment_execution/runs/$r/run.json | head -6; tail -12 experiment_execution/runs/$r/run.log | cut -c1-260; done
echo; tail -3 runs/e279_w_s0_p4/log.txt | cut -c1-240
pgrep -af "qgraph_train|run.py" | cut -c1-120
nvidia-smi --query-gpu=memory.used --format=csv,noheader
