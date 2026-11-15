#!/bin/bash
# E343 code step: the CPU tests of the evaluation folder that touch what jtest_eval.py imports or documents (the rest of run_all_tests.sh: see run_all_tests.log)
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="nice -n 10 bash tools/env/_py14.sh"
for t in "jtest_eval.py selftest" "test_jtest_eval.py" "qends_eval.py selftest" "jends_eval.py selftest" "jmaneuver.py selftest" "test_qgraph_eval.py" "test_qgraph_readout.py" "test_qgraph_score_glue.py" "test_tier_metrics.py" "test_zone_metrics.py" "vertex_eval.py selftest" "jrule_removal.py selftest" "jmodel_cost.py selftest"; do
  out=$($P $t 2>&1); rc=$?
  echo "$([ $rc -eq 0 ] && echo PASS || echo FAIL)  $t  | $(echo "$out" | tail -1 | cut -c1-120)"
done
