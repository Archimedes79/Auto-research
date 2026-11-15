#!/bin/bash
# scratch: commit F: the note of the schedule row t22.B2 (new arming after the reboot), own path, temporary index; refuses if the shared schedule.csv carries more than my one-line change
cd /mnt/c/Auto-research || exit 1
st=$(git diff --numstat -- bev_fusion/records/schedule.csv | cut -f1,2)
[ "$st" = "$(printf '1\t1')" ] || { echo "schedule.csv diff is not exactly my one line ($st): stop"; exit 1; }
P="bev_fusion/records/schedule.csv"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -3
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/commit_msg_F.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" $P 2>&1 | tail -5
git log --oneline -2
