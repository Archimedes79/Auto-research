#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/records/t25_p4_readout_2026-11-14.txt bev_fusion/experiments/t25_one_frame_one_scan/experiment.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 2
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/readout_msg.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-140
