#!/bin/bash
# scratch: commit the t23 second-seed unit (own paths; the ledger files differ from HEAD only by my four rows / notes, checked before)
cd /mnt/c/Auto-research || exit 1
B=bev_fusion
R=""
for d in t21.ctl_s1 t21.ctl_s1.eval t23.C_s1 t23.C_s1.eval; do
  for f in done run.json; do
    [ -f $B/records/runs/$d/$f ] && R="$R $B/records/runs/$d/$f"
  done
done
P="$B/experiments/t23_lane_line_loss/experiment.md $B/records/LEDGER.csv $B/records/LEDGER.md $B/records/night_2026-11-08.md $R"
echo "paths: $P" | tr ' ' '\n' | tail -n +2
git diff --stat -- $B/records/LEDGER.csv $B/records/LEDGER.md | cat
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F $B/temp/overview/commit_t23s1_msg.txt $P
