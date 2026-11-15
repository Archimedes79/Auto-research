#!/bin/bash
cd /mnt/c/Auto-research || exit 1
git fetch origin 2>&1 | tail -3
echo "behind/ahead: $(git rev-list --left-right --count origin/main...HEAD)"
git log --oneline origin/main..HEAD
echo "--- files in the unpushed commits"
git diff --name-status origin/main..HEAD
python3 autoresearch/tools/check_git.py > bev_fusion/temp/cleanup/check_git_final2.txt 2>&1
echo "check_git exit=$?"
grep -v 'pushing the tags would publish' bev_fusion/temp/cleanup/check_git_final2.txt | head -20
echo "tag-error lines: $(grep -c 'pushing the tags would publish' bev_fusion/temp/cleanup/check_git_final2.txt)"
