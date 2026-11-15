#!/bin/bash
# scratch: commit E: schedule row of t22.B2 + card status line, own paths, temporary index; refuses if the shared schedule.csv carries more than my one-line change
cd /mnt/c/Auto-research || exit 1
st=$(git diff --numstat -- bev_fusion/records/schedule.csv | cut -f1,2)
[ "$st" = "$(printf '1\t1')" ] || { echo "schedule.csv diff is not exactly my one line ($st): stop"; exit 1; }
cs=$(git diff --numstat -- bev_fusion/experiments/t22_student_encoder/experiment.md | cut -f1,2)
echo "card diff: $cs"
P="bev_fusion/records/schedule.csv bev_fusion/experiments/t22_student_encoder/experiment.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -3
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg_E.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" $P 2>&1 | tail -5
git log --oneline -2
