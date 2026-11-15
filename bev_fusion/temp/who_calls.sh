#!/bin/bash
# scratch: who still calls the planner's runner (runner/bev_runner.py) and the prototype (evaluation/bev_prototype.py) - tracked files only
cd /mnt/c/Auto-research || exit 1
echo "=== bev_runner (tracked py/sh/json files)"
git grep -n "bev_runner" -- '*.py' '*.sh' '*.json' ':!bev_fusion/runner/bev_runner.py' ':!bev_fusion/records/*' | cut -c1-185
echo; echo "=== bev_runner (tracked md files, count)"
git grep -c "bev_runner" -- '*.md' | head -20
echo; echo "=== bev_prototype (tracked py/sh/json files)"
git grep -n "bev_prototype" -- '*.py' '*.sh' '*.json' ':!bev_fusion/evaluation/bev_prototype.py' | cut -c1-185 | head -30
echo; echo "=== bev_prototype (tracked md files, count)"
git grep -c "bev_prototype" -- '*.md' | head -20
echo; echo "=== readers of objects.json / maps.npz (tracked py)"
git grep -ln "objects.json\|maps.npz" -- '*.py' ':!bev_fusion_lane_center/*' | head
echo; echo "=== prototype dirs in runs: $(ls -d bev_fusion/runs/*/prototype 2>/dev/null | wc -l)"
