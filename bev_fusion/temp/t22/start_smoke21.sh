#!/bin/bash
# scratch: the paired control of the stage-B smoke: t21_pair4 under the same smoke settings, no student; diagnostic, no card
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
exec python3 ../autoresearch/tools/run.py --id smoke.t21_pair4 --gpu --gpu-max-mb 3000 --split train --no-card -- bash temp/train/run_ar.sh temp/t22/smoke21/smoke_t21_pair4.json
