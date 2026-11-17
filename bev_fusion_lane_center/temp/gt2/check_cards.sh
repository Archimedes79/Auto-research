cd /mnt/c/Auto-research
for f in E309_pair-delivery-threshold E311_snap-connector-ends E307_narrow-end-heat E308_pair-head-with-node-tokens E310_maneuver-coverage-and-connector-headroom E313_heat-sigma-0.7; do
  python3 autoresearch/tools/experiment.py check bev_fusion_lane_center/experiments/$f/experiment.md 2>&1 | tail -n 1
done
