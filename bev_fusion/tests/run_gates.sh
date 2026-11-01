#!/usr/bin/env bash
# Run the synthetic gates of bev_fusion block by block and report pass / fail per block (plan.md section 10.1).
# Fast self-tests always; the training gates (flow, splat correction) and the slow self-tests of the chain (model, losses, trainer,
# autoGT) only with --all (they take minutes on the CPU).
# Usage (from the bev_fusion folder): wsl bash tests/run_gates.sh [--all]
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"     # autogt lives at the repository root since 2026-11-01 (`-m autogt selftest`)
PY="${PY:-$HOME/miniconda3/envs/gcar/bin/python}"
fail=0
run() {  # run <block> <command...>
  local name="$1"; shift
  local out
  if out=$("$@" 2>&1); then
    echo "PASS  $name  | $(echo "$out" | grep -v -i warn | tail -1 | cut -c1-140)"
  else
    echo "FAIL  $name  | $(echo "$out" | grep -v -i warn | tail -3 | tr '\n' ' ' | cut -c1-200)"
    fail=1
  fi
}
run "synthetic world"   "$PY" tests/synthetic_world.py
run "lift PD"           "$PY" training/lift_pd.py
run "camera encoder"    "$PY" training/camera_encoder.py
run "radar encoder"     "$PY" training/radar_bev.py
run "splat"             "$PY" training/splat.py
run "BEV trunk"         "$PY" training/bev_trunk.py
run "temporal fusion"   "$PY" training/temporal_fusion.py
run "fusion"            "$PY" training/fusion.py
run "timed heads"       "$PY" training/timed_head.py
run "object head"       "$PY" training/object_head.py
run "metrics"           "$PY" evaluation/metrics.py
run "front compare"     "$PY" evaluation/front_compare.py
run "paper fallbacks"   env OMP_NUM_THREADS=1 "$PY" training/fallbacks.py
run "oracle lift"       "$PY" tests/test_oracle_lift.py
run "Doppler"           "$PY" tests/test_doppler.py
run "GT cache, targets" "$PY" tests/test_gt_cache_targets.py
if [ -f runs/t10_dual_s0/ckpt_00003000.pt ]; then               # its G4 warm-starts from this real checkpoint (not in git)
  run "velocity hurdle" "$PY" tests/test_velocity_hurdle.py     # t11; CPU, about 1 min, 2.2 GB peak
else
  echo "SKIP  velocity hurdle  | runs/t10_dual_s0/ckpt_00003000.pt missing (G4 needs it)"
fi
run "slots, every kind" "$PY" tests/test_slots.py                   # five-layer build: contract test of every kind of every slot; CPU, no data, about 1 min
run "orientation"       "$PY" tests/test_orientation.py            # every drawer: x up, +y left, arrows, colour keys, overlays
run "nuScenes dataset"  "$PY" training/nuscenes_dataset.py          # nuScenes mini; about 20 s
run "mirror augment"    "$PY" training/augment.py                   # nuScenes mini; about 30 s
if [ "${1:-}" = "--all" ]; then
  run "whole chain"     "$PY" training/model.py                     # nuScenes mini, CPU; minutes
  run "losses"          "$PY" training/losses.py                    # CPU; minutes (overfit 100 steps)
  run "trainer"         "$PY" training/train.py                     # nuScenes mini + data/cache/cam_feats_mini, CPU; minutes
  run "autoGT"          "$PY" -m autogt selftest
  run "flow from frames" "$PY" tests/test_flow_from_frames.py
  run "flow correlation" "$PY" tests/test_flow_correlation.py
  run "projection (R-GEO)" "$PY" tests/test_projection_general.py
fi
exit $fail
