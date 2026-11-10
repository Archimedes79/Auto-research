#!/bin/bash
# scratch: the exact A1 command of the t22 card (small split, backbone frozen), through the launcher as a diagnostic (no card, never feeds a verdict)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
exec python3 ../autoresearch/tools/run.py --id t22.A1_s0 --gpu --gpu-max-mb 3000 --split train --no-card -- \
  bash -c '~/miniconda3/envs/gcar/bin/python -u training/distill_student.py --name t22_A1_s0 --split small --freeze-backbone --epochs 5 --batch 16 --lr 1e-3 --workers 4 >> runs/t22_A1_s0.log 2>&1'
