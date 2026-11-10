#!/bin/bash
# scratch: commit G (D65): the off config, its README row, the decision row, the schedule row of the L4 check; own paths, temporary index; refuses if a shared file carries more than my one added line
cd /mnt/c/Auto-research || exit 1
for f in bev_fusion/records/schedule.csv bev_fusion/training/configs/README.md bev_fusion_lane_center/experiment_execution/DECISIONS.md; do
  st=$(git diff --numstat -- $f | cut -f1,2)
  [ "$st" = "$(printf '1\t0')" ] || { echo "$f: diff is not exactly my one added line ($st): stop"; exit 1; }
done
P="bev_fusion/records/schedule.csv bev_fusion/training/configs/README.md bev_fusion/training/configs/t20_base_nol4.json bev_fusion_lane_center/experiment_execution/DECISIONS.md"
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -3
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t22/lane_off/commit_msg_G.txt --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" $P 2>&1 | tail -6
git log --oneline -2
