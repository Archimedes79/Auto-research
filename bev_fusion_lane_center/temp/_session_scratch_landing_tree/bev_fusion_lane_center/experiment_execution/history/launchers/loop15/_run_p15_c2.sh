#!/bin/bash
# E121d: consumers (route + ego, 3 seeds) on the refined graph with the route probabilities of the route head retrained on it (E121c)
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
rm -f $O/e121d.done
mkdir -p $O/e121_d
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_chain_data.py --out chain_r2 --tok tok_e120_a --route e121c > $O/e121_d/data.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p14_chain_consumer.py --data chain_r2 --out e121_d --modes route+ego --seeds 3 > $O/e121_d/run.log 2>&1
echo done > $O/e121d.done
