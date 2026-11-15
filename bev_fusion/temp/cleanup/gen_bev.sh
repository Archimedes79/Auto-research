#!/bin/bash
# BEV Fusion GT after the D83 merge: front v13 + lane l6 of the bev_fusion rig + packed masks (resumable)
cd /mnt/c/Auto-research || exit 1
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
s=$(date +%s)
nice -n 15 ~/miniconda3/envs/gcar/bin/python -m GtGenerator generate-bev ~/data/nuscenes_full --set full --workers 2 > /mnt/c/Auto-research/bev_fusion/temp/cleanup/gen_bev.log 2>&1
rc=$?
echo "exit=$rc after $(( $(date +%s) - s )) s" >> /mnt/c/Auto-research/bev_fusion/temp/cleanup/gen_bev.log
