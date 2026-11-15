#!/bin/bash
# keep the scripts of temp/e280 (E280 pre-check P11) in the git tag lcr/e280-scratch-1 through a temporary index (main is not touched)
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e280_scratch_index"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
find $L/temp/e280 -type f -name '*.py' -size -300k | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "E280 pre-check P11 scratch (scripts only: pair types, AUC, oracle step; main is not touched)")
git tag lcr/e280-scratch-1 "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -4
