#!/bin/bash
# one-off: keep the E276 scratch (scripts of temp/e276 and the top-level scripts of temp/gt_l4) in the git tag lcr/e276-scratch through a temporary index (main is not touched)
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e276_scratch_index"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
{ find $L/temp/e276 -type f \( -name '*.py' -o -name '*.sh' \) -size -300k; find temp/gt_l4 -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' \) -size -300k; } | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "E276 scratch, state of 2026-11-07 evening (scripts only: the GT l4 test run, the CPU pre-experiments P1-P4, the launcher chain, the tests runner; main is not touched)")
git tag -f lcr/e276-scratch "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -2
