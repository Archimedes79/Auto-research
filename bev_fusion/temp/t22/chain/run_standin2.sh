#!/bin/bash
# scratch: make the stand-in and twin run folders under temp/t22/chain and load both on the CPU (log: standin2.log)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=$HOME/miniconda3/envs/gcar/bin/python
rm -rf temp/t22/chain/standin_pair4 temp/t22/chain/twin_pair4
{
$PY temp/t22/chain/make_standin_run.py runs/t21_pair4_s0 temp/t22/chain/standin_pair4 --student runs/t22_A1_s0
$PY temp/t22/chain/make_standin_run.py runs/t21_pair4_s0 temp/t22/chain/twin_pair4 --twin
cat temp/t22/chain/standin_pair4/standin.json
$PY temp/t22/chain/standin_load.py temp/t22/chain/standin_pair4 runs/t21_pair4_s0 runs/t22_A1_s0/ckpt_00000710.pt
$PY temp/t22/chain/standin_load.py temp/t22/chain/twin_pair4 runs/t21_pair4_s0
$PY temp/t22/chain/make_standin_run.py runs/t21_pair4_s0 temp/t22/chain/twin_pair4 --check
} > temp/t22/chain/standin2.log 2>&1
echo done >> temp/t22/chain/standin2.log
