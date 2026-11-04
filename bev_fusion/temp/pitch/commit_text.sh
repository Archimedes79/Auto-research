#!/usr/bin/env bash
# commit own text records through a temporary index (the shared index belongs to every chat), move main with a checked update-ref, refresh the shared index entries of those paths
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MSG="$(cd "$(dirname "$0")" && pwd)/commit_msg.txt"
PATHS="bev_fusion_lane_center/reports/source/weekly/case_autoresearch_2026-11-01.md bev_fusion/experiments/t10_dual_features/experiment.md bev_fusion/experiments/t3_improve_loop/report.md bev_fusion/experiments/t6_soa_gap/report.md bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md bev_fusion/records/runs/pitch.maps_lfl bev_fusion/records/runs/pitch.maps_lfl_check"
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git add -- $PATHS
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p "$old" -F "$MSG")
unset GIT_INDEX_FILE
git update-ref -m "case v4 text records" refs/heads/main "$commit" "$old"
git reset -q HEAD -- $PATHS
git log --oneline -1
git show --stat --format= "$commit" | tail -12
