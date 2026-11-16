cd /mnt/c/Auto-research
for f in E298_driven-pair-weight E312_combined-end-maps E313_heat-sigma-0.7 E314_pairs-on-sigma-0.7-maps E304_far-range-weighted-maps; do
  python3 autoresearch/tools/experiment.py check bev_fusion_lane_center/experiments/$f/experiment.md 2>&1 | tail -n 1
done
python3 autoresearch/tools/commit_paths.py -m "E298 result (KEEP: driven pair recall +0.12/+0.11 on the E312 system), ledger notes" --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" bev_fusion_lane_center/experiments/E298_driven-pair-weight bev_fusion_lane_center/experiment_execution/LEDGER.csv bev_fusion_lane_center/experiment_execution/LEDGER.md 2>&1 | tail -n 2
