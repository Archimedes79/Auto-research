#!/usr/bin/env bash
# commit the fusion-area evaluation through a temporary index; run from anywhere: wsl bash bev_fusion/temp/pitch/commit_eval3.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git add -- bev_fusion/evaluation/bev_fusion_area.py bev_fusion/evaluation/bev_annotation_pts.py bev_fusion/evaluation/bev_response.py bev_fusion/evaluation/bev_mutants.py \
  bev_fusion/evaluation/bev_suite.py bev_fusion/evaluation/bev_task.py bev_fusion/evaluation/bev_gtcheck.py bev_fusion/evaluation/bev_selftest.py bev_fusion/tests/run_gates.sh
blob=$(git hash-object -w "$HERE/README_blob3.md")
git update-index --cacheinfo 100644,"$blob",bev_fusion/evaluation/README.md
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p "$old" -F "$HERE/commit_eval3_msg.txt")
unset GIT_INDEX_FILE
git update-ref -m "evaluation: fusion-area scoring" refs/heads/main "$commit" "$old"
git reset -q HEAD -- bev_fusion/evaluation/bev_fusion_area.py bev_fusion/evaluation/bev_annotation_pts.py bev_fusion/evaluation/bev_response.py bev_fusion/evaluation/bev_mutants.py \
  bev_fusion/evaluation/bev_suite.py bev_fusion/evaluation/bev_task.py bev_fusion/evaluation/bev_gtcheck.py bev_fusion/evaluation/bev_selftest.py bev_fusion/tests/run_gates.sh 2>/dev/null || true
git log --oneline -1
git show --stat --format= "$commit" | tail -14
