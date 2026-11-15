#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=~/miniconda3/envs/gcar/bin/python
$PY -c "import json; c=json.load(open('training/configs/t20_base_nol4.json')); print('json ok; fusion_area_only =', c['loss']['fusion_area_only'], '; lane_branch =', repr(c['model']['lane_branch']))"
OMP_NUM_THREADS=1 nice -n 15 $PY training/startable.py training/configs/t20_base_nol4.json 2>&1 | tail -n 6 | cut -c1-220
