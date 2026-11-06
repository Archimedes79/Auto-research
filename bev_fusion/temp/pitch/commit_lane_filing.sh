#!/usr/bin/env bash
# commit the filing of lane loops 19 - 22 (scratch left the tip of main, the scripts live in the tags lcr/loopNN-scratch) through a temporary index
# run from the repository root: wsl bash bev_fusion/temp/pitch/commit_lane_filing.sh
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
HERE="$(cd "$(dirname "$0")" && pwd)"
L=bev_fusion_lane_center
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git ls-files -- $L/temp/loop19 $L/temp/loop20 $L/temp/loop21 $L/temp/loop22 | xargs git rm -q --cached --
git add -- $L/training/data/prep/run_b456_gt.sh $L/data/rebuild.json $L/temp/README.md
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p "$old" -F "$HERE/commit_lane_filing_msg.txt")
unset GIT_INDEX_FILE
git update-ref -m "lane loops 19-22 filed" refs/heads/main "$commit" "$old"
git reset -q HEAD -- $L/data/rebuild.json $L/temp/README.md $L/training/data/prep/run_b456_gt.sh 2>/dev/null || true
git log --oneline -1
git show --stat --format= "$commit" | tail -4
