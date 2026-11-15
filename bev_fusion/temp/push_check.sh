#!/bin/bash
# scratch: fetch and the push check (read-only except the fetch)
cd /mnt/c/Auto-research || exit 1
git fetch origin 2>&1 | tail -3
echo "=== branch state"; git status -sb | head -1
echo "=== unpushed commits: $(git rev-list --count origin/main..HEAD)"
git log --format='%h %an %s' origin/main..HEAD | cut -c1-150 | head -60
echo "=== check_git.py"
python3 autoresearch/tools/check_git.py 2>&1 | tail -45 | cut -c1-200
