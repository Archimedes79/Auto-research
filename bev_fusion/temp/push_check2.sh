#!/bin/bash
cd /mnt/c/Auto-research || exit 1
python3 autoresearch/tools/check_git.py 2>&1 | grep -v "pushing the tags would publish forbidden file" | head -40 | cut -c1-220
echo "=== forbidden files in the four unpushed commits themselves:"
for c in $(git rev-list origin/main..HEAD); do git show --stat --format='%h' "$c" | tail -n +2 | head -0; done
git diff --name-status origin/main..HEAD | cut -c1-140 | head -40
echo "=== sizes > 1 MB in the unpushed range:"
git diff --name-only origin/main..HEAD | while read f; do [ -f "$f" ] && s=$(stat -c %s "$f") && [ "$s" -gt 1000000 ] && echo "$s $f"; done
echo "(none above means none)"
echo "=== tags not on the remote (count) and the ones that point into the unpushed range:"
git tag --contains origin/main 2>/dev/null | head -5
