#!/bin/bash
pgrep -af "launch_chain|tools/run.py" | cut -c1-110
ls /mnt/c/Auto-research/bev_fusion_lane_center/temp/e276 | grep -E "CHAIN_DONE|VERDICT"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
