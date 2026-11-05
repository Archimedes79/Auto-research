#!/usr/bin/env bash
# t11 vel cause: the CPU jobs one at a time. Wait until extract_all.sh has ended, then (1) the Doppler-at-boxes check, (2) the read-out probe
# of t1_r4_soa (positive control: a run whose velocity branch learned) with a within-file split.
# Run from bev_fusion/: bash temp/box_vel/after_extract.sh > temp/box_vel/out/after_extract.log 2>&1
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY=~/miniconda3/envs/gcar/bin/python
until grep -q "all done" temp/box_vel/out/extract_all.log; do sleep 20; done
avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
echo "=== $(date +%T) doppler_at_boxes (available ${avail} MB)"
if [ "$avail" -ge 3500 ]; then
  nice -n 10 $PY temp/box_vel/doppler_at_boxes.py runs/t8_ctl_s0 val 300 30 temp/box_vel/out/doppler_at_boxes_val.json || echo "FAILED doppler"
else echo "SKIPPED doppler (memory)"; fi
echo "=== $(date +%T) probe_fit t1_r4_soa (within-val split)"
if [ -s temp/box_vel/out/t1_r4_soa_val.npz ]; then
  nice -n 10 $PY temp/box_vel/probe_fit.py temp/box_vel/out/t1_r4_soa_val.npz temp/box_vel/out/t1_r4_soa_val.npz t1_r4_soa 2>&1 | grep -v Warning | tee temp/box_vel/out/probe_fit_t1_r4_soa.txt
else echo "no t1_r4_soa_val.npz"; fi
echo "=== $(date +%T) all done"
