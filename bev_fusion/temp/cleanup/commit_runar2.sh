#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P=bev_fusion/temp/train/run_ar.sh
git diff HEAD --stat -- $P | cat
ps aux | grep -E "run_ar\.sh|train\.py" | grep -v grep | cut -c1-120
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/runar_msg.txt $P
echo "rc=$?"
git log -1 --format='%h %s' | cut -c1-120
git show --stat --format= HEAD | cat
