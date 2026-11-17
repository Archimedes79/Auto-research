#!/bin/bash
# lane lead 2026-11-16: the clean tags (no forbidden file) that check_git.py lists, one per line, for an exact-list push
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/check_git.py 2>&1 | grep '^tags without a forbidden file' | sed 's/^tags without a forbidden file ([0-9]*): //' | tr ' ' '\n' | grep -v '^$' > bev_fusion_lane_center/temp/_session_scratch_lead/clean_tags.txt
wc -l bev_fusion_lane_center/temp/_session_scratch_lead/clean_tags.txt
