#!/bin/bash
# the real-data gates on the final tree (stored runs still evaluate): golden, door --real, bev_free, api --real; one after the other, only with enough free RAM
R=/mnt/c/Auto-research/bev_fusion
L=$R/temp/cleanup/real_gates
mkdir -p $L; : > $L/summary.txt
PY=$HOME/miniconda3/envs/gcar/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $R || exit 1
run() { local name=$1; shift
  while [ "$(free -m | awk '/^Mem:/{print $7}')" -lt 19000 ]; do echo "waiting for RAM ($(free -m | awk '/^Mem:/{print $7}') MB free)" >> $L/summary.txt; sleep 120; done
  local t0=$(date +%s); nice -n 10 "$@" > $L/$name.log 2>&1; local rc=$?
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s | $(grep -v -i warn $L/$name.log | tail -1 | cut -c1-160)" | tee -a $L/summary.txt; }
date +%T >> $L/summary.txt
run golden $PY evaluation/bev_selftest.py --golden
run door_real $PY tests/test_evaluate_door.py --real runs/t20_base_s0
run bev_free $PY tests/test_bev_free.py
run api_real $PY tests/test_evaluation_api.py --real runs/t20_base_s0
date +%T >> $L/summary.txt
