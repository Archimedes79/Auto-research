#!/bin/bash
python3 /mnt/c/Auto-research/bev_fusion_lane_center/temp/e276/reclaim.py
grep -E "MemFree|MemAvailable" /proc/meminfo
cat /proc/buddyinfo | head -4
