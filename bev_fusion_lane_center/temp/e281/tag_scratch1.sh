#!/bin/bash
# keep the scripts of temp/e281 (E281 pre-check P12) in the git tag lcr/e281-scratch-2 through a temporary index (main is not touched)
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e281_scratch_index2"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
find $L/temp/e281 -type f \( -name '*.py' -o -name '*.sh' \) -size -300k | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "E281 pre-check P12 and threshold check P13 scratch (scripts only; main is not touched)")
git tag lcr/e281-scratch-2 "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -6
