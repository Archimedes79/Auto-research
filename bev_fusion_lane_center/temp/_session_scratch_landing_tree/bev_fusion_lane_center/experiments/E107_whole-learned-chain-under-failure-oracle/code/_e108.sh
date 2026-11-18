#!/bin/bash
# Loop 12 / E108: oracle hold (upper bound of the memory hold) and gradual onset. Chain, run in the background.
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
mkdir -p $O/e108_a $O/e108_ramp8
for p in dark blur cam_front3_off radar_off; do
  for v in single hold_mon ema_hold_mon_0.5; do
    cp -n $O/e105_a/graphs_${p}_${v}.pkl $O/e108_a/ 2>/dev/null
    cp -n $O/e105_a/comps_${p}_${v}.npy $O/e108_a/ 2>/dev/null
  done
done
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
bash _pyr.sh p11_temporal.py --perts dark,blur,cam_front3_off,radar_off --variants hold_oracle,ema_hold_oracle:0.5 --out e108_a > $O/e108_a/run.log 2>&1
bash _pyr.sh p11_temporal.py --perts dark,blur --variants single,ema_hold_mon:0.5,ema_hold_oracle:0.5 --out e108_ramp8 --ramp 8 > $O/e108_ramp8/run.log 2>&1
echo CHAIN_DONE > $O/e108_chain.done
