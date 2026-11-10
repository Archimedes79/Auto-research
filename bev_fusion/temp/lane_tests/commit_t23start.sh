#!/bin/bash
# scratch: commit the t23 start records (own hunks of two files) with a message file
cd "$(dirname "$0")/../../.." || exit 1
P="bev_fusion/records/schedule.csv bev_fusion/experiments/t23_lane_line_loss/experiment.md"
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/lane_tests/commit_t23start_msg.txt $P
