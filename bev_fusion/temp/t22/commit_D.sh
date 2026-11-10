#!/bin/bash
# scratch: commit D: gate C0 records (card, ledger csv + md, schedule), own paths, through the temporary index
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/experiments/t22_student_encoder/experiment.md bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md bev_fusion/records/schedule.csv"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -4
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg_D.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" $P 2>&1 | tail -6
git log --oneline -3
