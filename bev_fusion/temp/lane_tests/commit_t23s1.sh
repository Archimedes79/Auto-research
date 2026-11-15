#!/bin/bash
# scratch: commit the card section 4 (second seed: prediction and rule before the runs) and the schedule row
cd "$(dirname "$0")/../../.." || exit 1
P="bev_fusion/experiments/t23_lane_line_loss/experiment.md bev_fusion/records/schedule.csv"
git diff --stat -- $P
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/lane_tests/commit_t23s1_msg.txt $P
