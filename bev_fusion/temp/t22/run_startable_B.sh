#!/bin/bash
# scratch: startable check of the stage-B config after the split key is resolved (what `trainsuite start` hands to the trainer)
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
mkdir -p temp/t22/cfg
for n in t22_B_s0 t21_pair4; do
  $PY training/splits.py resolve < training/configs/$n.json > temp/t22/cfg/$n.json || exit 1
  echo "=== $n"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY training/startable.py temp/t22/cfg/$n.json 2>&1 | tail -25 | cut -c1-250
done
