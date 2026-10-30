#!/bin/bash
ps -eo pid,etime,pcpu,args | grep -E "[p]ython" | grep -v "lane_center\|p16_train\|run.py" | cut -c1-150
echo "--- gpu"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
