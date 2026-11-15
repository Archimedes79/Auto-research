#!/bin/bash
# scratch: commit J: the card section "Seed twin and R arm" (postponed, D68) + three configs + their README rows, own paths, temporary index; "dry" as $1 = dry run
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/experiments/t22_student_encoder/experiment.md bev_fusion/training/configs/README.md bev_fusion/training/configs/t21_pair4_s1.json bev_fusion/training/configs/t22_B_s1.json bev_fusion/training/configs/t22_R_s0.json"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -8
echo "--- check_git exit: ${PIPESTATUS[0]}"
EXTRA=""
[ "$1" = dry ] && EXTRA="--dry-run"
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg_J.txt $P $EXTRA 2>&1 | tail -10
git log --oneline -2 | cut -c1-160
