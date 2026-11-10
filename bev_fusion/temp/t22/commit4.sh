#!/bin/bash
# scratch: commit of the card and this chat's ledger rows through a temporary index (dry run first when $1 = dry)
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/experiments/t22_student_encoder/experiment.md bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -15
echo "--- check_git exit: ${PIPESTATUS[0]}"
EXTRA=""
[ "$1" = dry ] && EXTRA="--dry-run"
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg4.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" bev_fusion/experiments/t22_student_encoder/experiment.md \
  --blob bev_fusion/records/LEDGER.csv=bev_fusion/temp/t22/blob_LEDGER.csv --blob bev_fusion/records/LEDGER.md=bev_fusion/temp/t22/blob_LEDGER.md $EXTRA 2>&1 | tail -25
