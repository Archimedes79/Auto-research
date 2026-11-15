#!/bin/bash
# scratch: commit the t24 unit (own paths; tests/README.md as a blob = HEAD + my row, because another chat has an uncommitted hunk in it)
cd /mnt/c/Auto-research || exit 1
B=bev_fusion
P="$B/training/lane_branch.py $B/training/model.py $B/tests/test_lane_pull.py $B/training/configs/t24_lane_pull.json $B/training/configs/README.md $B/training/README.md $B/experiments/t24_lane_pull/experiment.md $B/records/schedule.csv"
git diff --stat -- $B/training/lane_branch.py $B/training/model.py $B/records/schedule.csv | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F $B/temp/overview/commit_t24_msg.txt $P --blob $B/tests/README.md=/mnt/c/Auto-research/$B/temp/overview/tests_readme_blob.md
