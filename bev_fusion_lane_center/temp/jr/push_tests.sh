#!/bin/bash
# CPU-only selftests of the code that changed since origin/main (run before the push of 2026-11-14)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P="nice -n 10 bash tools/env/_py14.sh"
fail=0
run() { name="$1"; shift; out=$($P "$@" 2>&1); rc=$?; echo "$([ $rc -eq 0 ] && echo PASS || echo FAIL)  $name | $(echo "$out" | tail -1 | cut -c1-110)"; [ $rc -eq 0 ] || fail=1; }
run "gpu_gate" evaluation/tests/test_gpu_gate.py
run "jrule_removal" evaluation/jrule_removal.py selftest
run "jmodel_cost" evaluation/jmodel_cost.py selftest
run "jprune_cost" evaluation/jprune_cost.py selftest
run "jprune_probe" evaluation/jprune_probe.py selftest
run "jmaps_simplify" training/grid/test_jmaps_simplify.py
run "mnms_layer" training/grid/test_mnms_layer.py
run "jwire" training/grid/jwire.py
run "jwire_head" training/grid/jwire_head.py
echo "FAIL=$fail"
