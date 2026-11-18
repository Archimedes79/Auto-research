#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gcar
source /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/bflc_env.sh
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
echo "=== B3 loi on D3"
python p16_train.py --design D3 --name smoke_flags --smoke --eval none --workers 2 --set ver_mode=\"loi\" 2>&1 | tail -2 | cut -c1-300
echo "=== B1 geo link on D7"
python p16_train.py --design D7 --name smoke_flags --smoke --eval none --workers 2 --set link_mode=\"geo\" 2>&1 | tail -2 | cut -c1-300
