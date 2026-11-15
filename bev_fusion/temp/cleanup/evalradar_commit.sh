#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/evaluation/bev_adapter.py bev_fusion/evaluation/bev_modules.py bev_fusion/evaluation/bev_probe.py bev_fusion/evaluation/bev_selftest.py bev_fusion/evaluation/README.md bev_fusion/experiments/t25_one_frame_one_scan/experiment.md"
git diff --stat -- $P | tail -n 8
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 3
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/evalradar_msg.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-140
