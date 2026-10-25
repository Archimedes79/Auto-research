#!/bin/bash
# Loop 13 / E114: the whole learned chain (trace network -> tokens -> cross-attention route head) under failure with the spatial hold, and the route confidence of the chain
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
mkdir -p $O/e114_tok $O/e114_a
rm -f $O/e114.done
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
bash _pyr.sh p11_temporal.py --perts dark,blur,cam_front3_off --variants single,ema_hold_mon:0.5,maxdecay:0.95 --out e114_tok --tokens > $O/e114_tok/run.log 2>&1
bash _pyr.sh p13_route_fail.py --dir e114_tok --out e114_a --a_list 0 > $O/e114_a/run.log 2>&1
echo done > $O/e114.done
