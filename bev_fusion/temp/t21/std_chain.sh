#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
PY="$HOME/miniconda3/envs/gcar/bin/python"
for kind in "recurrent 3" "pair 2" "window 3"; do
  nice -n 15 bash temp/t21/memwatch.sh 2300 "$PY" temp/t21/std_chain.py $kind 2>&1 | grep -v -i warn | tail -3
done
