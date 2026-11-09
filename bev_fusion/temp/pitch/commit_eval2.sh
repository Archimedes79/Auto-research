#!/usr/bin/env bash
# commit the evaluation work of 2026-11-01 (curve metrics, GT checks, response test, mutation check, generalisation, data trace) through a temporary index
# run from the repository root: wsl bash bev_fusion/temp/pitch/commit_eval2.sh
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
HERE="$(cd "$(dirname "$0")" && pwd)"
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git add -- autoresearch/tools/run_report.py bev_fusion/evaluation/bev_curves.py bev_fusion/evaluation/bev_gtcheck.py bev_fusion/evaluation/bev_response.py \
  bev_fusion/evaluation/bev_mutants.py bev_fusion/evaluation/bev_suite.py bev_fusion/evaluation/bev_task.py
python3 "$HERE/readme_blobs2.py"
python3 "$HERE/gates_blob.py"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p "$old" -F "$HERE/commit_eval2_msg.txt")
unset GIT_INDEX_FILE
git update-ref -m "evaluation: curves, GT checks, response test" refs/heads/main "$commit" "$old"
git reset -q HEAD -- autoresearch/tools/run_report.py bev_fusion/evaluation bev_fusion/tests/run_gates.sh 2>/dev/null || true
git log --oneline -1
git show --stat --format= "$commit" | tail -12
