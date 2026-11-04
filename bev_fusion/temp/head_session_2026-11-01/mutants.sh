#!/usr/bin/env bash
# six mutants of training/losses.py; every one must fail `losses.py --fusion-area` (CPU, seconds each, no optimiser step)
cd /mnt/c/Auto-research/bev_fusion/temp/head_session_2026-11-01 || exit 1
PY=~/miniconda3/envs/gcar/bin/python
SRC=../../training/losses.py
mkdir -p m
n=0; killed=0
mut() {  # mut <name> <sed expression>
  n=$((n+1))
  sed "$2" $SRC > m/losses.py
  if cmp -s $SRC m/losses.py; then echo "MUTANT $1: sed changed nothing (bad mutant)"; return; fi
  if PYTHONPATH=/mnt/c/Auto-research/bev_fusion/training OMP_NUM_THREADS=2 nice -n 10 $PY m/losses.py --fusion-area > m/out.log 2>&1; then
    echo "MUTANT $1: SURVIVED"
  else
    killed=$((killed+1)); echo "MUTANT $1: killed ($(grep -E 'AssertionError' m/out.log | head -1 | cut -c1-110))"
  fi
}
mut off          's/    if cfg.fusion_area_only:  /    if False:  /'
mut obj_only     's/        w_map, w_obj = w_map \* fusion, w_obj \* fusion/        w_obj = w_obj * fusion/'
mut map_only     's/        w_map, w_obj = w_map \* fusion, w_obj \* fusion/        w_map = w_map * fusion/'
mut default_on   's/    fusion_area_only: bool = False /    fusion_area_only: bool = True  /'
mut cam_masked   's/"w_cam": w_cam, "w_radar": radar,/"w_cam": w_cam * fusion, "w_radar": radar,/'
mut radar_masked 's/"w_cam": w_cam, "w_radar": radar,/"w_cam": w_cam, "w_radar": radar * fusion,/'
echo "killed $killed of $n"
rm -f m/losses.py m/out.log
