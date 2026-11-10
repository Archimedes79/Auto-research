#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/check_git.py 2>&1 | grep -v "^  pushing" | tail -14 | cut -c1-200
git status -sb | head -1
