#!/bin/bash
cd /mnt/c/Auto-research || exit 1
git fetch origin 2>&1 | tail -3
echo "behind/ahead: $(git rev-list --left-right --count origin/main...HEAD)"
git log --oneline origin/main..HEAD
echo "--- files"
git diff --name-status origin/main..HEAD
echo "--- GtGenerator selftest in an empty cwd"
PY=$HOME/miniconda3/envs/gcar/bin/python
E=/mnt/c/Auto-research/bev_fusion/temp/cleanup/empty_cwd
rm -rf "$E"; mkdir -p "$E"
( cd "$E" && PYTHONPATH=/mnt/c/Auto-research timeout 600 $PY -m GtGenerator selftest 2>&1 | tail -4; echo "rc=${PIPESTATUS[0]}"; echo "left in cwd: $(ls -A | wc -l) entries: $(ls -A | tr '\n' ' ')" )
python3 autoresearch/tools/check_git.py > bev_fusion/temp/cleanup/check_git_final3.txt 2>&1
echo "check_git non-tag lines:"; grep -v 'pushing the tags would publish' bev_fusion/temp/cleanup/check_git_final3.txt | grep -E 'ERROR|ahead|to push|staged' | head
