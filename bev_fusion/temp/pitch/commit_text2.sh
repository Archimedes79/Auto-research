#!/usr/bin/env bash
# commit own text records through a temporary index (the shared index belongs to every chat), move main with a checked update-ref, refresh the shared index entries of those paths
# run from the repository root: wsl bash bev_fusion/temp/pitch/commit_text2.sh
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MSG="$(cd "$(dirname "$0")" && pwd)/commit_msg2.txt"
PATHS="bev_fusion_lane_center/reports/source/weekly/case_autoresearch_2026-11-01.md bev_fusion/experiments/t10_dual_features/experiment.md bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md"
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git add -- $PATHS
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p "$old" -F "$MSG")
unset GIT_INDEX_FILE
git update-ref -m "t10 long run stopped, text records" refs/heads/main "$commit" "$old"
git reset -q HEAD -- $PATHS
git log --oneline -1
git show --stat --format= "$commit" | tail -8
