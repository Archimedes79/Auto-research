#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bev
python /mnt/c/ML-Experiment/bev_fusion_lane_center/training/data/prep/centerline_gt.py \
  --dataroot "$HOME/data/nuscenes" --version v1.0-mini \
  --out /mnt/c/ML-Experiment/out/centerline_gt --debug 8 2>&1 | grep -v -i -E 'warning|np\.|deprecat' | tail -20
