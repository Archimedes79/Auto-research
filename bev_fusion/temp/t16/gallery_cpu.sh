#!/bin/bash
# scratch: the lane-divider gallery of the baseline: the overview picture of the chosen frames (worst / median / best lane-divider IoU of different scenes,
# temp/t16/lane_frames_t16_full_s0.txt) on the CPU, after the video job has finished (one python job at a time, 2 threads, low priority)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
PY=/home/ml/miniconda3/envs/gcar/bin/python
while pgrep -f "bev_suite.py runs/t16_full_s0 --device cpu.*qual videos" > /dev/null; do sleep 20; done
date +%T
for lab in $(grep "^PICK" temp/t16/lane_frames_t16_full_s0.txt | awk '{print $3}'); do
  scene=${lab%/*}; idx=$((10#${lab#*/}))
  echo "== $scene/$idx $(date +%T)"
  nice -n 15 $PY evaluation/bev_suite.py runs/t16_full_s0 --device cpu --workers 1 --out runs/t16_full_s0/eval/showcase qual frame $scene/$idx --view overview --history warmup 2>&1 | tail -3 | cut -c1-200
done
date +%T
