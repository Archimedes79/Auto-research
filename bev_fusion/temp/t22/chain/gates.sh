#!/bin/bash
# scratch: the existing gates that touch the files of the stage-B chain step, one at a time, 2 threads (log: temp/t22/chain/gates.log)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$(cd .. && pwd)"
PY=$HOME/miniconda3/envs/gcar/bin/python
run() { local name="$1"; shift; local t0=$(date +%s); if out=$("$@" 2>&1); then echo "PASS  $name ($(( $(date +%s) - t0 )) s) | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-110)"; else echo "FAIL  $name ($(( $(date +%s) - t0 )) s) | $(echo "$out" | grep -v -i warn | tail -3 | tr '\n' ' ' | cut -c1-260)"; fi; }
run "bundle"              $PY tests/test_bundle.py
run "frozen frame"        $PY tests/test_frozen.py
run "interface"           $PY training/interface.py
run "dual camera geometry" $PY tests/test_dual_camera_geometry.py
run "camera features"     $PY tests/test_camera_features.py
run "lane branch, dual"   $PY tests/test_lane_branch_dual.py
run "lane branch, chain"  $PY tests/test_lane_branch_chain.py
run "lr groups"           $PY tests/test_lr_groups.py
run "speed flags"         $PY tests/test_speed_flags.py
run "hygiene"             $PY tests/test_hygiene.py
run "motion joint"        $PY tests/test_motion_joint.py
run "freeze, probe"       $PY tests/test_freeze_probe.py
run "launch saving"       $PY tests/test_launch_saving.py
run "host batch"          $PY tests/test_host_batch.py
run "stream temporal"     $PY tests/test_stream_temporal.py
run "trainkit equivalence" $PY tests/test_trainkit_equivalence.py
run "startable"           $PY training/startable.py
run "student encoder"     $PY tests/test_student_encoder.py
run "student chain"       $PY tests/test_student_chain.py
echo "done"
