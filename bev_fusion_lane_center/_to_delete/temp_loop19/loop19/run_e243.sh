#!/bin/bash
# E243 timing replicate: six configurations, four rounds, round-robin (a c b d e f), a fresh process per replicate, 200 timed frames.
# Needs an idle machine (GPU and CPU): the assembly is CPU numba. Writes runs/e18/latency/latency_*_e243_<arm>_r<k>.json (the tool's own files)
# and a flat json per replicate in runs/e19/e243/ (ms_per_frame, asm_ms, points) for the analysis in temp/loop19/analyse_e243.py.
cd "$(dirname "$0")/../.."
P="bash tools/env/_py14.sh"
mkdir -p runs/e19/e243
declare -A CFG=(
  [a]="--name e123_D7_r3_j1 --M 192 --K0 96 --rec 0 --fork_prune 0.3"
  [c]="--name e123_D7_r3_f1 --M 288 --K0 144 --rec 0 --fork_prune 0.3"
  [b]="--name e18_D7_j1eq --M 192 --K0 96 --rec 0 --fork_prune 0.4"
  [d]="--name e18_D7_j1rc --M 192 --K0 96 --rec 0 --fork_prune 0.4"
  [e]="--name e123_D7_r3_rc1 --M 288 --K0 144 --rec 0 --fork_prune 0.4"
  [f]="--name e123_D7_r3_f1 --M 288 --K0 144 --rec 1 --fork_prune 0.3"
)
busy() { [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)" -gt 5 ] || nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
for k in 1 2 3 4; do
  for arm in a c b d e f; do
    for try in 1 2 3; do busy && { echo "round $k arm $arm: GPU busy, wait 20 s"; sleep 20; } || break; done
    busy && { echo "round $k arm $arm: GPU still busy after 60 s, replicate void (kill rule)"; echo "{\"void\": true}" > runs/e19/e243/${arm}_r${k}.json; continue; }
    union=2
    $P p18_latency.py ${CFG[$arm]} --n 200 --union $union --tag _e243_${arm}_r${k} > runs/e19/e243/${arm}_r${k}.log 2>&1 || { echo "round $k arm $arm FAILED"; tail -n 5 runs/e19/e243/${arm}_r${k}.log; continue; }
    f=$(ls -t runs/e18/latency/*_e243_${arm}_r${k}.json | head -1)
    python3 - "$f" "runs/e19/e243/${arm}_r${k}.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
json.dump(dict(arm=sys.argv[1].split("_e243_")[1][0], ms_per_frame=r["total"]["median_ms"], asm_ms=r["assembly"]["median_ms"], builder_ms=r["builder"]["median_ms"],
               points=r["points_per_frame_mean"], src=sys.argv[1]), open(sys.argv[2], "w"))
EOF
    echo "round $k arm $arm: $(cat runs/e19/e243/${arm}_r${k}.json)"
  done
done
echo "E243 replicates done"
