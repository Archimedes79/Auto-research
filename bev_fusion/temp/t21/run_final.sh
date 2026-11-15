#!/usr/bin/env bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
PY="$HOME/miniconda3/envs/gcar/bin/python"
run() {
  s=$(date +%s)
  if out=$(nice -n 15 "$@" 2>&1); then r=PASS; else r=FAIL; fi
  echo "$r $* $(( $(date +%s) - s )) s | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-170)"
}
run "$PY" temp/t21/golden.py check
run "$PY" tests/test_frozen.py
run "$PY" training/startable.py
run "$PY" tests/test_stream_temporal.py
run "$PY" tests/test_slots.py
run "$PY" tests/test_motion_joint.py
run "$PY" tests/test_lane_branch_dual.py
run "$PY" tests/test_layers.py
run "$PY" tests/test_bundle.py
run "$PY" training/temporal_fusion.py
run "$PY" training/motion_block.py
run "$PY" training/timed_head.py
run "$PY" tests/test_trainkit_equivalence.py
