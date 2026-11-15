#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion_lane_center/experiment_execution/decisions/D91.md bev_fusion/studies/radar_sweep_density.py bev_fusion/studies/README.md bev_fusion/records/radar_sweep_density_2026-11-13.txt"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 2
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/d91_msg2.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-120
