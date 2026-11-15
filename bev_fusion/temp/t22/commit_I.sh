#!/bin/bash
# scratch: commit I: D65 check records (DECISIONS row, schedule note, my two ledger rows from blobs built on the current HEAD), own paths, temporary index; "dry" as $1 = dry run
cd /mnt/c/Auto-research || exit 1
for f in bev_fusion_lane_center/experiment_execution/DECISIONS.md bev_fusion/records/schedule.csv; do
  st=$(git diff --numstat -- $f | cut -f1,2)
  [ "$st" = "$(printf '1\t1')" ] || { echo "$f diff is not exactly my one line ($st): stop"; exit 1; }
done
bash bev_fusion/temp/t22/run_build_blob.sh 2>&1 | head -4 | cut -c1-260
P="bev_fusion_lane_center/experiment_execution/DECISIONS.md bev_fusion/records/schedule.csv bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -8
echo "--- check_git exit: ${PIPESTATUS[0]}"
EXTRA=""
[ "$1" = dry ] && EXTRA="--dry-run"
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg_I.txt bev_fusion_lane_center/experiment_execution/DECISIONS.md bev_fusion/records/schedule.csv \
  --blob bev_fusion/records/LEDGER.csv=bev_fusion/temp/t22/blob_LEDGER.csv --blob bev_fusion/records/LEDGER.md=bev_fusion/temp/t22/blob_LEDGER.md $EXTRA 2>&1 | tail -10
git log --oneline -2
