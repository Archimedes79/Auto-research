#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/guard_hook.py --selftest 2>&1 | tail -n 3 | cut -c1-220
echo "--- check bev champion / bev project.json / a free file / lane PROBLEM.md:"
for f in bev_fusion/records/champion.json bev_fusion/project.json bev_fusion/README.md bev_fusion_lane_center/PROBLEM.md; do
  printf '%s -> ' "$f"; python3 autoresearch/tools/guard_hook.py check "$f" 2>&1 | head -n 1 | cut -c1-160
done
