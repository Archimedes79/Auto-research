#!/usr/bin/env bash
# t11 box velocity: the feature extractions of probe_features.py, one after another on the CPU (2 threads, nice; light devkit, ~2.4 GB).
# A job is skipped when less than 3.5 GB are available at its start (other chats' jobs come first).
# Run from bev_fusion/: bash temp/box_vel/extract_all.sh > temp/box_vel/out/extract_all.log 2>&1
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY=~/miniconda3/envs/gcar/bin/python
mkdir -p temp/box_vel/out
for job in "runs/t8_ctl_s0 val 300 t8_ctl_s0_val 30" "runs/t8_ctl_s0 train 300 t8_ctl_s0_train 40" "runs/t1_r4_soa val 200 t1_r4_soa_val 30" \
           "runs/t1_r4_soa train 300 t1_r4_soa_train 40"; do
  set -- $job
  avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
  echo "=== $(date +%T) $1 $2 $3 frames, $5 scenes (available ${avail} MB)"
  if [ "$avail" -lt 3500 ]; then echo "SKIPPED (memory) $job"; continue; fi
  [ -f "temp/box_vel/out/$4.npz" ] && { echo "exists, kept"; continue; }
  nice -n 10 $PY temp/box_vel/probe_features.py "$1" "$2" "$3" "temp/box_vel/out/$4.npz" "$5" || echo "FAILED $job"
done
echo "=== $(date +%T) all done"
