#!/bin/bash
# scratch: commit H: card + schedule notes + my three ledger rows (blobs built from the current HEAD), own paths, temporary index; "dry" as $1 = dry run
cd /mnt/c/Auto-research || exit 1
st=$(git diff --numstat -- bev_fusion/records/schedule.csv | cut -f1,2)
[ "$st" = "$(printf '2\t2')" ] || { echo "schedule.csv diff is not exactly my two lines ($st): stop"; exit 1; }
P="bev_fusion/experiments/t22_student_encoder/experiment.md bev_fusion/records/schedule.csv bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -15
echo "--- check_git exit: ${PIPESTATUS[0]}"
EXTRA=""
[ "$1" = dry ] && EXTRA="--dry-run"
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg_H.txt bev_fusion/experiments/t22_student_encoder/experiment.md bev_fusion/records/schedule.csv \
  --blob bev_fusion/records/LEDGER.csv=bev_fusion/temp/t22/blob_LEDGER.csv --blob bev_fusion/records/LEDGER.md=bev_fusion/temp/t22/blob_LEDGER.md $EXTRA 2>&1 | tail -25
git log --oneline -2
