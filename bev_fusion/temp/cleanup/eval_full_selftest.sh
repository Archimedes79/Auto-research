#!/usr/bin/env bash
# scratch: the whole evaluation selftest suite (nice, 2 threads)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="${PY:-$HOME/miniconda3/envs/gcar/bin/python}"
nice -n 15 "$PY" evaluation/bev_modules.py --selftest 2>&1 | grep -v -i warn | tail -2; echo "modules rc=${PIPESTATUS[0]}"
nice -n 15 "$PY" evaluation/bev_selftest.py 2>&1 | grep -v -i warn | grep -v "^ok " | tail -15; echo "selftest rc=${PIPESTATUS[0]}"
