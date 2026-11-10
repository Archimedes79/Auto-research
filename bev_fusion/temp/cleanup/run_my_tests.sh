#!/bin/bash
# scratch (Image Encoder, clean-base round): the tests and self-tests of my area, CPU, 2 threads, one line per test; usage: run_my_tests.sh LOGFILE
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY="$HOME/miniconda3/envs/gcar/bin/python"
LOG=${1:-temp/cleanup/my_tests.log}
: > $LOG
run() {
  local name="$1"; shift
  local t0=$(date +%s)
  local out; out=$("$@" 2>&1); local rc=$?
  echo "$( [ $rc = 0 ] && echo PASS || echo FAIL )  $name  ($(( $(date +%s) - t0 )) s)  | $(echo "$out" | grep -v -i warn | tail -2 | tr '\n' ' ' | cut -c1-230)" >> $LOG
  [ $rc = 0 ] || echo "$out" | tail -25 >> $LOG.fail
}
rm -f $LOG.fail
run "student chain"       $PY tests/test_student_chain.py
run "student encoder"     $PY tests/test_student_encoder.py
run "stream tools"        $PY tests/test_stream_tools.py
run "startable selftest"  $PY training/startable.py
run "interface selftest"  $PY training/interface.py
run "lane_branch selftest" $PY training/lane_branch.py
run "feature_cache selftest" $PY training/feature_cache.py
run "model selftest"      $PY training/model.py
run "module probe"        $PY evaluation/bev_modules.py --selftest
run "lane branch chain"   $PY tests/test_lane_branch_chain.py
run "lane branch dual"    $PY tests/test_lane_branch_dual.py
run "lane pull"           $PY tests/test_lane_pull.py
run "camera features"     $PY tests/test_camera_features.py
run "dual camera geometry" $PY tests/test_dual_camera_geometry.py
run "oracle lift"         $PY tests/test_oracle_lift.py
run "lift PD"             $PY training/lift_pd.py
run "camera encoder"      $PY training/camera_encoder.py
run "paper fallbacks"     $PY training/fallbacks.py
echo "ALL DONE $(date +%H:%M:%S)" >> $LOG
