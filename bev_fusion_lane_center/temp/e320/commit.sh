#!/bin/bash
cd /mnt/c/Auto-research
B=bev_fusion_lane_center
P="$B/training/grid/jends.py $B/training/train/jends_train.py $B/evaluation/jends_eval.py $B/evaluation/qends_eval.py $B/evaluation/jmaneuver.py $B/training/grid/README.md $B/evaluation/README.md $B/experiments/E320_soft-attach-replaces-snap/experiment.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 3 || exit 1
python3 autoresearch/tools/commit_paths.py -F $B/temp/e320/commit_msg.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" $P 2>&1 | tail -n 12
