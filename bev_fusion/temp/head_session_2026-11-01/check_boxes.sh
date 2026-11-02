#!/usr/bin/env bash
# checks of the box unification, CPU, low priority, NO optimiser step anywhere (object_head's refiner self-test trains: not run)
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=~/miniconda3/envs/gcar/bin/python
export OMP_NUM_THREADS=2
run() { local name=$1; shift; if out=$(nice -n 10 "$@" 2>&1); then echo "PASS  $name | $(echo "$out" | tail -1 | cut -c1-140)"; else echo "FAIL  $name | $(echo "$out" | tail -5 | tr '\n' ' ' | cut -c1-400)"; fi; }
run "extract selftest"            $PY training/extract.py
run "object head (decode part)"   $PY -c "import sys; sys.path.insert(0,'training'); import object_head as O; O._selftest()"
run "slots, every kind"           $PY tests/test_slots.py
run "model selftest"              $PY training/model.py
$PY temp/head_session_2026-11-01/bench_boxes.py 2>&1 | tail -4
