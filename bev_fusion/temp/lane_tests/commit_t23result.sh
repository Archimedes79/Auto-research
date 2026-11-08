#!/bin/bash
# scratch: commit the t23 result records (own paths; the shared files show only my hunks, checked by git diff --stat)
cd "$(dirname "$0")/../../.." || exit 1
P="bev_fusion/experiments/t23_lane_line_loss/experiment.md bev_fusion/records/schedule.csv bev_fusion/records/status_2026-11-07.md bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md bev_fusion/records/runs/t23.C_s0 bev_fusion/records/runs/t23.C_s0.eval"
git diff --stat -- $P
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/lane_tests/commit_t23result_msg.txt $P
