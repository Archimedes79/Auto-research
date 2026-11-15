#!/bin/bash
# scratch: commit the closure of t24 (own paths: the card and the schedule; both were clean in git before my edit)
cd /mnt/c/Auto-research || exit 1
B=bev_fusion
P="$B/experiments/t24_lane_pull/experiment.md $B/records/schedule.csv"
git diff --stat -- $P | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F $B/temp/overview/commit_t24_close_msg.txt $P
