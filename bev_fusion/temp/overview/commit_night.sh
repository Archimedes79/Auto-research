#!/bin/bash
# scratch: commit the night record and the timestamp fix (own paths; card and schedule were committed by me and are otherwise clean)
cd /mnt/c/Auto-research || exit 1
B=bev_fusion
P="$B/records/night_2026-11-08.md $B/experiments/t24_lane_pull/experiment.md $B/records/schedule.csv"
git diff --stat -- $B/experiments/t24_lane_pull/experiment.md $B/records/schedule.csv | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F $B/temp/overview/commit_night_msg.txt $P
