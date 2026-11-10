#!/bin/bash
# scratch: the stand-in with student_frames 'all' on 4 real val frames on the CPU (data path end to end: dataset -> collate -> device move -> model -> metrics); no GPU
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
N=runs/smoke_t22_all_cpu
[ -d $N ] || $PY temp/t22/chain/make_standin_run.py runs/t21_pair4_s0 $N --student runs/t22_A2_s0 --frames all || exit 1
nice -n 12 $PY -u evaluation/evaluate.py $N --quant --n 4 --device cpu --workers 0 --ckpt-dir runs/t21_pair4_s0 > temp/t22/smoke_all_cpu.log 2>&1
echo "exit $?"
tail -12 temp/t22/smoke_all_cpu.log | cut -c1-260
