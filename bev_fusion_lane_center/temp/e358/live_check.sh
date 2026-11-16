#!/bin/bash
# E358 live check on attached ends (CPU, 24 val_clean frames per seed): the records of the LIVE pass against the stored-dump records (acceptance in the card section 2: all fields equal except at most one pair near p = 0.1 in one frame, atol 2e-3)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
for S in 0 1; do
  nice -n 10 bash tools/env/_py14.sh jwire_check.py live --seed $S --pair-run e324_s_s$S --max 24 --workers 2 --device cpu --ends att > temp/e358/live_s$S.log 2>&1
  echo "live seed $S exit $? $(date +%H:%M:%S)" >> temp/e358/live_done.txt
done
echo "LIVE DONE" >> temp/e358/live_done.txt
