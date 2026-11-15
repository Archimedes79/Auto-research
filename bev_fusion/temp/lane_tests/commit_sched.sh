#!/bin/bash
# scratch: commit the schedule row of t23 (own row of a shared record file; the working tree equals HEAD plus this one line)
cd "$(dirname "$0")/../../.." || exit 1
git diff --stat -- bev_fusion/records/schedule.csv
python3 autoresearch/tools/check_git.py --paths bev_fusion/records/schedule.csv || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/lane_tests/commit_sched_msg.txt bev_fusion/records/schedule.csv
