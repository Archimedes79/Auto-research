#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
date +%H:%M:%S; cat /proc/loadavg; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; free -g | sed -n 2p
ps -eo pid,etime,stat,pcpu,cmd | grep -E "run\.py --id|jmaps_train|jmaps_eval|jends_train|jtiming|jprune" | grep -v grep | grep -v "bash " | cut -c1-140
for r in e347_g_s0 e347_o_s0; do echo "$r: $(grep -c . runs/$r/train.log 2>/dev/null) lines; $(grep -E 'step [0-9]+' runs/$r/train.log 2>/dev/null | tail -1 | cut -c1-150)"; done
