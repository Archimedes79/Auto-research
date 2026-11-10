#!/bin/bash
# keep the E276 / E277 / E279 scratch (text files of temp/e276 and temp/e275, scripts only) in the git tag lcr/e276-scratch-2 through a temporary index (main is not touched)
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e276_scratch_index2"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
find $L/temp/e276 $L/temp/e275 -type f \( -name '*.py' -o -name '*.sh' \) -size -300k | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "E276 / E277 / E278 / E279 scratch, state of 2026-11-08 04:00 (scripts only: scheduler, start scripts, analyses, report builder, ledger note scripts; main is not touched)")
git tag lcr/e276-scratch-2 "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -2
