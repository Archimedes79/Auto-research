#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/evaluation/bev_free.py bev_fusion/tests/test_bev_free.py"
git status --short -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_free3_msg.txt $P
git status --short -- $P | cat
