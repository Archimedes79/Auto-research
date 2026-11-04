#!/bin/bash
# Mirko 2026-11-03 'Yes, time it': two diagnostic runs (no verdict) through run.py, each alone on the GPU: timing.phase_a (fresh init = first 1000 steps of a training) and timing.phase_c
# (warm start from the trained t20.base_s0 checkpoint = last 1000 steps). The machine lock (exclusive, taken only while the GPU is idle) keeps other launchers away for the ~15 minutes.
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion AR_LOCK_OWNER=tools-timing
for i in $(seq 1 720); do                                      # up to 6 h for an idle GPU
  python3 ../autoresearch/tools/machine_lock.py take tools-timing --hours 1 --exclusive --idle --note "tools chat: phase timing of the t20 step (Mirko's go 2026-11-03)" && break
  sleep 30
done
python3 ../autoresearch/tools/machine_lock.py show | grep -q tools-timing || { echo "no lock taken"; exit 1; }
trap 'python3 ../autoresearch/tools/machine_lock.py release tools-timing' EXIT
python3 ../autoresearch/tools/run.py --id timing.phase_a --no-card --no-git --gpu --gpu-max-mb 2500 --wait-gpu 120 --split train --ram-mb 12000 --timeout 40 \
  -- bash temp/speed_tools/phase_run.sh a
python3 ../autoresearch/tools/run.py --id timing.phase_c --no-card --no-git --gpu --gpu-max-mb 2500 --wait-gpu 120 --split train --ram-mb 12000 --timeout 40 \
  -- bash temp/speed_tools/phase_run.sh c runs/t20_base_s0/ckpt_00006000.pt
