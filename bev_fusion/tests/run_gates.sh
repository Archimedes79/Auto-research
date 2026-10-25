#!/usr/bin/env bash
# Run the synthetic gates of bev_fusion block by block and report pass / fail per block (plan.md section 10.1).
# Fast self-tests always; the training gates (flow, splat correction) only with --all (they take minutes on the CPU).
# Usage (from the bev_fusion folder): wsl bash tests/run_gates.sh [--all]
cd "$(dirname "$0")/.." || exit 1
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
run "oracle lift"       "$PY" tests/test_oracle_lift.py
run "Doppler"           "$PY" tests/test_doppler.py
if [ "${1:-}" = "--all" ]; then
  run "flow from frames" "$PY" tests/test_flow_from_frames.py
fi
exit $fail
