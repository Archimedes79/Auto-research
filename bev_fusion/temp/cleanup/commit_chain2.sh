#!/bin/bash
cd /mnt/c/Auto-research || exit 1
AR_PROJECT=../bev_fusion python3 autoresearch/tools/check_structure.py 2>&1 | tail -3
python3 autoresearch/tools/check_git.py --paths bev_fusion/README.md bev_fusion/records/jobs 2>&1 | tail -3
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/chain_msg2.txt bev_fusion/README.md bev_fusion/records/jobs/chain/README.md 2>&1 | tail -6
git fetch origin 2>&1 | tail -2
echo "behind/ahead: $(git rev-list --left-right --count origin/main...HEAD)"
git log --oneline -3
