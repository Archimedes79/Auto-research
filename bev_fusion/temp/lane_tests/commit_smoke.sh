#!/bin/bash
# scratch: commit the t23 card status and the smoke run record (own paths only)
cd "$(dirname "$0")/../../.." || exit 1
P="bev_fusion/experiments/t23_lane_line_loss/experiment.md bev_fusion/records/runs/smoke.t23_C/run.json bev_fusion/records/runs/smoke.t23_C/NOTE.txt"
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/lane_tests/commit_smoke_msg.txt $P
