#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/chain_msg.txt bev_fusion/README.md bev_fusion/records/jobs/README.md bev_fusion/records/jobs/chain 2>&1 | tail -15
git log --oneline -3
