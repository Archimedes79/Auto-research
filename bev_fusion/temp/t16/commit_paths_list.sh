#!/bin/bash
# scratch: the paths of the commit "t17 / t18 results, the report, timing" (own text records only), then check_git.py --paths on them
cd /mnt/c/Auto-research || exit 1
R=bev_fusion/records/runs
P="bev_fusion/README.md bev_fusion/experiments/t18_motion_lr/experiment.md bev_fusion/experiments/t16_full_network/experiment.md bev_fusion/records/LEDGER.csv bev_fusion/records/LEDGER.md bev_fusion/records/open_findings_2026-11-02.md"
for d in t16.small_bg1_s0 t16.small_clip_s0 t16.small_ctl_s0 t16.small_final_s0 t16.small_s0 t16.smoke_s0 t16.full_s0 t16.full_s0.eval t17.feat t17.data t17.data.eval t18.smoke t18.fix t18.fix.eval showcase.gpu timing.idle3 timing.idle3b timing.idle3c timing.idle3d; do
  for f in run.json done; do
    [ -f "$R/$d/$f" ] && P="$P $R/$d/$f"
  done
done
echo "$P" | tr ' ' '\n' > bev_fusion/temp/t16/commit_paths.txt
wc -l bev_fusion/temp/t16/commit_paths.txt
/home/ml/miniconda3/envs/gcar/bin/python autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 15
