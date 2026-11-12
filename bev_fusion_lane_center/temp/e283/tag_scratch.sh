#!/bin/bash
# keep the scripts of temp/e283 (system report 2026-11-08) in the git tag lcr/system-report-scratch-1 through a temporary index (main is not touched)
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e283_scratch_index"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
find $L/temp/e283 -type f \( -name '*.py' -o -name '*.sh' \) -size -300k | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "system report 2026-11-08 scratch (builder, asset launcher, helpers; main is not touched)")
git tag lcr/system-report-scratch-1 "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -3
