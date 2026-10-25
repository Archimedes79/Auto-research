#!/usr/bin/env bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bev
cd "$HOME/work/bevfusion"
export PYTHONWARNINGS=ignore
python /mnt/c/ML-Experiment/bev_fusion_lane_center/training/data/prep/prep_graph.py "$HOME/exp/graph" 2>&1 | grep -v -i -E 'warn|np\.|deprecat|Loaded|=====|^Done|seconds'
