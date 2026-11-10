#!/bin/bash
# scratch: the 20-step smoke of stage B2 (end-to-end proof before the 3 h run); diagnostic through the launcher, no card
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
exec python3 ../autoresearch/tools/run.py --id smoke.t22_B2_s0 --gpu --gpu-max-mb 3000 --split train --no-card -- bash temp/train/run_ar.sh temp/t22/smoke/smoke_t22_B2_s0.json
