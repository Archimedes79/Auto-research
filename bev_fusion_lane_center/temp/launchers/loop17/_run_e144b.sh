#!/bin/bash
# E144b: the consumer behind the map builder B with NO map for the builder (and the route head) and with a 2 m map error for the builder
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/logs
until [ -f $L/e144.done ]; do sleep 20; done
SERB=""
for p in dark blur cam_front3_off; do SERB="$SERB,${p}_single,${p}_maxdecay_0.95,${p}_gateB"; done
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_consumer.py --out e144_map_none --mapnet 1 --ckpt e17_map_b --map_mode none --series ${SERB#,} > $L/e144_map_none.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_consumer.py --out e144_map_m2 --mapnet 1 --ckpt e17_map_b --map_mode exact --map_sigma 2 --series ${SERB#,} > $L/e144_map_m2.log 2>&1
echo finished > $L/e144b.done
