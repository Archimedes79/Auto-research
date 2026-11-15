#!/bin/bash
# scratch: the stage-B smoke run again, with the student's backbone recomputed in the backward (memory); diagnostic, no card
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
exec python3 ../autoresearch/tools/run.py --id smoke.t22_B_s0c --gpu --gpu-max-mb 3000 --split train --no-card -- bash temp/train/run_ar.sh temp/t22/smoke/smoke_t22_B_s0c.json
