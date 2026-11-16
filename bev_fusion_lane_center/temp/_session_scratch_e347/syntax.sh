#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate gcar
cd /mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e347/stage
for f in $(find . -name '*.py'); do python -c "import ast,sys; ast.parse(open('$f', encoding='utf-8').read())" && echo "ok $f"; done
