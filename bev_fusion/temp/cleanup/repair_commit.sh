#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion_lane_center/experiment_execution/DECISIONS.md"
F=bev_fusion/temp/cleanup/DECISIONS_fixed.md
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/repair_msg.txt --blob $P=$F || exit 1
cp $F $P || exit 1
git status --short -- $P | cat
echo "rows with D68 / D67 in HEAD:"
git show HEAD:$P | grep -c -E '^\| 2026-11-09 \| D6[78] '
