#!/usr/bin/env bash
# commit the cleanup of 2026-11-01 (files left the tip of main; they stay in history) and the new run_report tool through a temporary index:
# own paths only, README rows as HEAD + my edit only (the working copies carry other chats' uncommitted lines)
# run from the repository root: wsl bash bev_fusion/temp/pitch/commit_cleanup.sh
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
HERE="$(cd "$(dirname "$0")" && pwd)"
MSG="$HERE/commit_cleanup_msg.txt"
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git rm -q --cached -- bev_fusion/tests/box_optimality_fresh.txt bev_fusion/tests/run_when_gpu_free.sh \
  bev_fusion_lane_center/evaluation/loops/loop16/p16_diag.py bev_fusion_lane_center/evaluation/loops/loop16/p16_junc_head_eval.py
git add -- autoresearch/tools/run_report.py bev_fusion/evaluation/bev_suite.py
python3 "$HERE/readme_blobs.py"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p "$old" -F "$MSG")
unset GIT_INDEX_FILE
git update-ref -m "cleanup 2026-11-01, run_report" refs/heads/main "$commit" "$old"
git reset -q HEAD -- autoresearch/tools/run_report.py bev_fusion/evaluation/bev_suite.py autoresearch/README.md bev_fusion/evaluation/README.md bev_fusion/tests/README.md 2>/dev/null || true
git log --oneline -1
git show --stat --format= "$commit" | tail -10
