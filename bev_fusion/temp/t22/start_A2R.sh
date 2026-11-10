#!/bin/bash
# scratch: the A2 command of the t22 card with the ResNet-18 backbone (large split, all weights, backbone at 0.1 x the learning rate), through the launcher as a diagnostic (no card, never feeds a verdict)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
exec python3 ../autoresearch/tools/run.py --id t22.A2R_s0 --gpu --gpu-max-mb 3000 --split train --no-card -- \
  bash -c '~/miniconda3/envs/gcar/bin/python -u training/distill_student.py --name t22_A2R_s0 --backbone resnet18 --split large --epochs 6 --batch 16 --lr 1e-3 --backbone-mult 0.1 --workers 4 >> runs/t22_A2R_s0.log 2>&1'
