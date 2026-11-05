#!/bin/bash
# scratch: the mutation run of the stage-B chain step (log: temp/t22/chain/mutate_chain.log)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
$HOME/miniconda3/envs/gcar/bin/python temp/t22/mutate_chain.py "$@" > temp/t22/chain/mutate_chain.log 2>&1
