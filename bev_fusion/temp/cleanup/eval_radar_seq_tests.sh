#!/usr/bin/env bash
# scratch: the evaluation selftests that touch the radar_seq perturbations (nice, 2 threads)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="${PY:-$HOME/miniconda3/envs/gcar/bin/python}"
echo "== bev_modules selftest"
nice -n 15 "$PY" evaluation/bev_modules.py --selftest 2>&1 | grep -v -i warn | tail -5; echo "rc=${PIPESTATUS[0]}"
echo "== bev_selftest probe check"
nice -n 15 "$PY" evaluation/bev_selftest.py --only "bev_probe" 2>&1 | grep -v -i warn | tail -8; echo "rc=${PIPESTATUS[0]}"
