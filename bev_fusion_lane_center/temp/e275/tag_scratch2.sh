#!/bin/bash
# one-off: keep the E275 scratch (second tag, after the close-out) (text files of temp/e275) in the git tag lcr/e275-scratch-2 through a temporary index (main is not touched)
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e275_scratch_index2"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
find $L/temp/e275 -type f \( -name '*.py' -o -name '*.sh' \) -size -300k | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "E275 scratch 2, state of 2026-11-08 00:40 after the scoring pass and the close-out (scripts only: launcher, analysis, summary, ledger note scripts; main is not touched)")
git tag lcr/e275-scratch-2 "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -2
