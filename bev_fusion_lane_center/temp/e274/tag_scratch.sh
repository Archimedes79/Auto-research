#!/bin/bash
# one-off: keep the E274 scratch (text files of temp/e274) in the git tag lcr/e274-scratch through a temporary index (main is not touched)
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e273_scratch_index"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
find $L/temp/e274 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.txt' -o -name '*.log' -o -name '*.html' -o -name '*.json' \) -size -300k | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "E274 scratch (diagnostics of the node misses, read-out sweeps, pixel layer scores, launchers; main is not touched)")
git tag lcr/e274-scratch "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -2
