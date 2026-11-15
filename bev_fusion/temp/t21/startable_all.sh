#!/usr/bin/env bash
# the split name is resolved the way `trainsuite start` does (training/splits.py resolve); the startable check reads the composed file
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
PY="$HOME/miniconda3/envs/gcar/bin/python"
mkdir -p temp/t21/composed
for c in pair pair4 rec ctl; do
  echo "== t21_$c"
  "$PY" training/splits.py resolve < training/configs/t21_$c.json > temp/t21/composed/t21_$c.json
  nice -n 15 "$PY" training/startable.py temp/t21/composed/t21_$c.json 2>&1 | grep -v -i warn | cut -c1-330
done
