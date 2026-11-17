#!/bin/bash
# lane lead 2026-11-16: commit the finished metric cards of runs/eval that the committed cards cite (small json text records, 345 KB in total, 42 paths)
cd /mnt/c/Auto-research || exit 1
mapfile -t P < <(tr -d '\r' < bev_fusion_lane_center/temp/_session_scratch_lead/evalcards_paths.txt | grep -v '^$')
echo "paths: ${#P[@]}"
python3 autoresearch/tools/commit_paths.py -F bev_fusion_lane_center/temp/_session_scratch_lead/msg_evalcards.txt "${P[@]}" 2>&1 | tail -n 4
