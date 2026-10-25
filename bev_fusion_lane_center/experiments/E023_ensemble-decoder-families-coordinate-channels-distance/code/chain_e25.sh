#!/usr/bin/env bash
# E25 (FCN mit/ohne Koordinatenkanaele): startet erst nach dem CNT-Training (GPU-Teilung bremst sonst beide), dann Felder, Schaetzer-Auswertung, Fehleranalyse.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/out
until [ -f $L/crn/b123_cnt/heatnet.pt ]; do sleep 20; done
bash $S/run_crn_e25.sh > $L/crn_e25.log 2>&1
cd $S
NET_COORD=0 ~/run_py.sh crn_fields.py fcn8k small 48 $L/crn/b123_fcn8k/heatnet.pt
NET_COORD=1 ~/run_py.sh crn_fields.py fcncoord8k small 48 $L/crn/b123_fcncoord8k/heatnet.pt
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/loop01_foundation_runs/run_post_all.sh fcn8k est > $L/crn_post_fcn8k.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/loop01_foundation_runs/run_post_all.sh fcncoord8k est > $L/crn_post_fcncoord8k.log 2>&1
RANGE_TAG=_e25 ~/run_py.sh exp_range.py lidar_unet64 fcn8k fcncoord8k > $L/crn_range_e25.log 2>&1
echo "E25 ALLES ENDE" > $L/crn_e25_done.log
