#!/bin/bash
# t26.ref_front_base: the front-sector detection table (AP at 0.5 / 1 / 2 / 4 m, mAP2, range bins) of CRN (reference, first) against the base on the 1620 frames of the 43 common scenes, with the paired scene bootstrap; CPU
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
CRNPKL=$HOME/work/CRN/outputs/t26_crn/crn_front.pkl
mkdir -p temp/t26/front_base
nice -n 15 ~/miniconda3/envs/gcar/bin/python evaluation/bev_suite.py front crn=$CRNPKL temp/t26/base_run --n all --x-max 50 --boot 1000 --out temp/t26/front_base/front_crn_vs_base.md
