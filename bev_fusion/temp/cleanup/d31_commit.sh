#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/evaluation/evaluate.py bev_fusion/evaluation/bev_suite.py bev_fusion/evaluation/bev_eval.py bev_fusion/evaluation/bev_task.py bev_fusion/evaluation/bev_selftest.py bev_fusion/evaluation/README.md bev_fusion/tests/test_evaluate_door.py bev_fusion/tests/test_evaluation_api.py bev_fusion/records/frame_changes.md"
git diff --stat -- $P | tail -n 12 | cut -c1-100
git status --short bev_fusion/evaluation | cat
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/d31_msg.txt $P || exit 1
git log --format='%h %s' -1 | cut -c1-120
git fetch origin --quiet; echo "behind/ahead: $(git rev-list --left-right --count origin/main...HEAD)"
