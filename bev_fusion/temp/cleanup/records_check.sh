#!/bin/bash
# the untracked run records and result packages of bev_fusion: do they pass the git rule?
cd /mnt/c/Auto-research || exit 1
git status --porcelain -uall -- bev_fusion/records bev_fusion/results | grep '^??' | cut -c4- | grep -v '^bev_fusion/results/BEV_' > bev_fusion/temp/cleanup/untracked_records.txt
echo "untracked files: $(wc -l < bev_fusion/temp/cleanup/untracked_records.txt)"
sed 's|.*/||' bev_fusion/temp/cleanup/untracked_records.txt | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -12
python3 autoresearch/tools/check_git.py --paths $(cat bev_fusion/temp/cleanup/untracked_records.txt | tr '\n' ' ') 2>&1 | tail -15
