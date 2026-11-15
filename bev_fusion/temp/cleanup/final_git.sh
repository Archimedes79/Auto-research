#!/bin/bash
cd /mnt/c/Auto-research || exit 1
git fetch origin 2>&1 | tail -3
echo "behind/ahead: $(git rev-list --left-right --count origin/main...HEAD)"
python3 autoresearch/tools/check_git.py > bev_fusion/temp/cleanup/check_git_final.txt 2>&1
echo "check_git exit=$?"
head -60 bev_fusion/temp/cleanup/check_git_final.txt
