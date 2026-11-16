#!/bin/bash
# one-off: keep the E274 scratch (text files of temp/e274) in the git tag lcr/e274-scratch-2 through a temporary index (main is not touched)
set -e
cd "$(git rev-parse --show-toplevel)"
L=bev_fusion_lane_center
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/e274_scratch2_index"
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
find $L/temp/e274 -type f \( -name '*.py' -o -name '*.sh' \) -size -300k | sort -u > "$GIT_INDEX_FILE.list"
wc -l < "$GIT_INDEX_FILE.list"
xargs -d '\n' git add -f -- < "$GIT_INDEX_FILE.list"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p HEAD -m "E274 scratch, state of 2026-11-05 evening (scripts only: diagnostics, read-out sweeps, launchers, ledger note script; main is not touched)")
git tag lcr/e274-scratch-2 "$commit"
rm -f "$GIT_INDEX_FILE" "$GIT_INDEX_FILE.list"
git show --stat --format='%h %s' "$commit" | tail -2
