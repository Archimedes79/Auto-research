#!/bin/bash
# scratch: the 20-step smoke run of the stage-B config (student trainable, anchor on) through the launcher as a diagnostic (no card, never feeds a verdict)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
exec python3 ../autoresearch/tools/run.py --id smoke.t22_B_s0 --gpu --gpu-max-mb 3000 --split train --no-card -- bash temp/train/run_ar.sh temp/t22/smoke/smoke_t22_B_s0.json
