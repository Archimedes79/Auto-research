#!/bin/bash
for p in $(pgrep -f "[t]est_flow_explains.py"); do ps -o pid,etime,rss,args -p $p | tail -1 | cut -c1-140; done
free -g | head -2
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
