#!/bin/bash
cd /mnt/c/Auto-research || exit 1
git fetch origin 2>&1 | tail -3
echo "--- ahead / behind origin/main:"
git rev-list --left-right --count origin/main...HEAD
echo "--- unpushed commits:"
git log origin/main..HEAD --format='%h %an %s' | cut -c1-150
echo "--- check_git:"
python3 autoresearch/tools/check_git.py 2>&1 | tail -30
