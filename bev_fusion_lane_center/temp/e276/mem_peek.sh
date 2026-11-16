#!/bin/bash
date +%H:%M
free -m
dmesg 2>/dev/null | tail -5
ps -eo pid,etime,rss,args --sort=-rss | head -8 | cut -c1-170
ls /mnt/c/Auto-research/bev_fusion_lane_center/runs | grep -E "e279" 
ls -la /mnt/c/Auto-research/bev_fusion_lane_center/runs/e279_w_s0_p4 | head
