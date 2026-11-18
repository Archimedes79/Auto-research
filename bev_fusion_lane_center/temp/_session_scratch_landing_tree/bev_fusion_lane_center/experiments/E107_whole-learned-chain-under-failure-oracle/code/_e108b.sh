#!/bin/bash
# Loop 12 / E108b: the heat monitor with a longer reference window (16 unflagged frames instead of 4) - gradual onset and abrupt onset, EMA 0.5 + hold; waits for the E108 chain
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
while [ ! -f $O/e108_chain.done ]; do sleep 20; done
mkdir -p $O/e108_ramp8_rn16 $O/e108_a_rn16
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
bash _pyr.sh p11_temporal.py --perts dark,blur --variants ema_hold_mon:0.5 --out e108_ramp8_rn16 --ramp 8 --ref_n 16 > $O/e108_ramp8_rn16/run.log 2>&1
bash _pyr.sh p11_temporal.py --perts dark,blur --variants ema_hold_mon:0.5 --out e108_a_rn16 --ref_n 16 > $O/e108_a_rn16/run.log 2>&1
echo done > $O/e108b.done
