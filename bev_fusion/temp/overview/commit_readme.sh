#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/records/README.md"
git diff --stat -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/overview/commit_readme_msg.txt $P
