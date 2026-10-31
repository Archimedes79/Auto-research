#!/bin/bash
date; uptime
echo "--- training processes"
ps -eo pid,lstart,etime,args | grep -E "[t]rain\.py|[c]hain_r" | grep -v "lane_center" | cut -c1-160 | head -8
echo "--- gpu"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- runs"; ls -la --time-style=+%H:%M runs/ 2>/dev/null
for d in runs/*/; do echo "== $d"; ls --time-style=+%H:%M -la $d | tail -4; tail -2 $d/log.jsonl 2>/dev/null | cut -c1-300; done
