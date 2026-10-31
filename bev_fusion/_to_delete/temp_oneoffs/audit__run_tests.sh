#!/bin/bash
export OMP_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for f in training/model.py training/train.py training/bev_trunk.py training/augment.py; do
  echo "== $f"; $PY $f 2>&1 | grep -v -i warn | tail -3 | cut -c1-400
done
