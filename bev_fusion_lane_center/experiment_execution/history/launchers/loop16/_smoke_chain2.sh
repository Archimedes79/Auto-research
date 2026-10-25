#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gcar
source /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/bflc_env.sh
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
python p16_chain_data.py --tok tok_p16_smoke --route p16_route_smoke --out p16chain_smoke --draws 1 --workers 2 2>&1 | tail -8
python p14_chain_consumer.py --data p16chain_smoke --out p16_cons_smoke --epochs 2 --seeds 1 2>&1 | tail -8
