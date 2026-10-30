#!/bin/bash
date; uptime; free -g | head -2
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- python jobs"
ps -eo pid,etime,pcpu,args | grep -E "[p]ython" | cut -c1-140 | head
