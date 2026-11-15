#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
{ $PY temp/t22/chain/old_vs_new.py new; $PY temp/t22/chain/old_vs_new.py old; $PY temp/t22/chain/old_vs_new.py compare; } > temp/t22/chain/ovn.log 2>&1
echo done >> temp/t22/chain/ovn.log
