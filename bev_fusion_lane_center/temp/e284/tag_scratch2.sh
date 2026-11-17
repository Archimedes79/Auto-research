#!/bin/bash
# keep the scripts of temp/e284 (junction / error analysis of the lane head, 2026-11-08 / 09) in the git tag lcr/e284-scratch-2 through a temporary index (main is not touched)
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e284_scratch_index2"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
find $L/temp/e284 -type f \( -name '*.py' -o -name '*.sh' \) -size -300k | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "E283 junction / error analysis of the lane head: scratch scripts of 2026-11-08 / 09 / 10 (p17 end offset added) (scripts only; main is not touched)")
git tag lcr/e284-scratch-2 "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -6
