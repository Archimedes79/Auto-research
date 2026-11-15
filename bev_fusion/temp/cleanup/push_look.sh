#!/bin/bash
cd /mnt/c/Auto-research || exit 1
git fetch origin --quiet 2>&1 | tail -n 2
echo "behind/ahead: $(git rev-list --left-right --count origin/main...HEAD)"
echo "== unpushed commits:"
git log origin/main..HEAD --format='%h %an %ad %s' --date=format:'%H:%M' | cut -c1-150
echo "== files added or changed by them, by top folder (count):"
git diff --name-only origin/main..HEAD | cut -d/ -f1-2 | sort | uniq -c | sort -rn | head -n 14
echo "== added files that are big or binary-looking:"
git diff --diff-filter=A --name-only origin/main..HEAD | grep -E '\.(npz|pt|pth|ckpt|png|jpg|mp4|pdf|pptx|zip|bundle|svg|html)$' | head -n 10
echo "== check_git:"
python3 autoresearch/tools/check_git.py 2>&1 | tail -n 14 | cut -c1-200
