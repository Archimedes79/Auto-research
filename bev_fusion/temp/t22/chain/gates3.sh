#!/bin/bash
# scratch: the existing gates that touch the files of the stage-B chain step, one at a time, 2 threads (log: temp/t22/chain/gates3.log)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$(cd .. && pwd)"
PY=$HOME/miniconda3/envs/gcar/bin/python
run() { local name="$1"; shift; local t0=$(date +%s); if out=$("$@" 2>&1); then echo "PASS  $name ($(( $(date +%s) - t0 )) s) | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-110)"; else echo "FAIL  $name ($(( $(date +%s) - t0 )) s) | $(echo "$out" | grep -v -i warn | tail -3 | tr '\n' ' ' | cut -c1-260)"; fi; }
run "student chain"       $PY tests/test_student_chain.py
run "student encoder"     $PY tests/test_student_encoder.py
run "slots"               $PY tests/test_slots.py
run "bundle"              $PY tests/test_bundle.py
run "lane branch, dual"   $PY tests/test_lane_branch_dual.py
run "lane branch, chain"  $PY tests/test_lane_branch_chain.py
run "host batch"          $PY tests/test_host_batch.py
run "stream temporal"     $PY tests/test_stream_temporal.py
run "startable"           $PY training/startable.py
run "trainkit equivalence" $PY tests/test_trainkit_equivalence.py
run "velocity hurdle"     $PY tests/test_velocity_hurdle.py
run "camera features"     $PY tests/test_camera_features.py
run "dual camera geometry" $PY tests/test_dual_camera_geometry.py
run "frozen frame"        $PY tests/test_frozen.py
echo "done"
