#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md bev_fusion/records/schedule.csv"
git diff --stat -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_close_msg.txt $P
