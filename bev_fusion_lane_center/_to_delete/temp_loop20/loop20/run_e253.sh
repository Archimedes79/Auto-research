#!/bin/bash
# E253 (E243 repeated with a machine guard): six configurations, six rounds, round-robin (a c b d e f), a fresh process per replicate, 200 timed frames,
# then at most two repair rounds for configurations with fewer than six valid replicates (referee: voided replicates are repeated).
# Needs an idle machine (GPU and CPU): the assembly is CPU numba. Writes runs/e18/latency/latency_*_e253_<arm>_r<k>.json (the tool's own files),
# a flat json per replicate in runs/e19/e253/ (ms_per_frame, asm_ms, points) and a machine sidecar m_<arm>_r<k>.txt (load average, Windows CPU %, GPU clock / temperature).
# Guards (a replicate is void and skipped when one fails after 60 s of waiting): GPU utilisation > 5 % or another GPU process; WSL 1-minute load average > 3;
# Windows total CPU utilisation (2 s sample through powershell.exe) > 20 %. Analysis: temp/loop20/analyse_e253.py.
cd "$(dirname "$0")/../.."
P="bash tools/env/_py14.sh"
D=runs/e19/e253
mkdir -p $D
declare -A CFG=(
  [a]="--name e123_D7_r3_j1 --M 192 --K0 96 --rec 0 --fork_prune 0.3"
  [c]="--name e123_D7_r3_f1 --M 288 --K0 144 --rec 0 --fork_prune 0.3"
  [b]="--name e18_D7_j1eq --M 192 --K0 96 --rec 0 --fork_prune 0.4"
  [d]="--name e18_D7_j1rc --M 192 --K0 96 --rec 0 --fork_prune 0.4"
  [e]="--name e123_D7_r3_rc1 --M 288 --K0 144 --rec 0 --fork_prune 0.4"
  [f]="--name e123_D7_r3_f1 --M 288 --K0 144 --rec 1 --fork_prune 0.3"
)
wincpu() { powershell.exe -NoProfile -Command "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object Name -eq '_Total').PercentProcessorTime" 2>/dev/null | tr -d '\r\n'; }
busy() {
  [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)" -gt 5 ] && return 0
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && return 0
  awk '{exit !($1 > 3.0)}' /proc/loadavg && return 0
  w=$(wincpu); [ -n "$w" ] && [ "$w" -gt 20 ] && return 0
  return 1
}
nvalid() { n=0; for f in $D/${1}_r*.json; do [ -f "$f" ] && ! grep -q void "$f" && n=$((n+1)); done; echo $n; }
rep() {  # arm k
  arm=$1; k=$2
  for try in 1 2 3; do busy && { echo "round $k arm $arm: machine busy, wait 20 s"; sleep 20; } || break; done
  busy && { echo "round $k arm $arm: machine still busy after 60 s, replicate void"; echo "{\"void\": true}" > $D/${arm}_r${k}.json; return; }
  { date +%T; cat /proc/loadavg; echo "win_cpu_pct=$(wincpu)"; nvidia-smi --query-gpu=clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader; } > $D/m_${arm}_r${k}.txt
  $P p18_latency.py ${CFG[$arm]} --n 200 --union 2 --tag _e253_${arm}_r${k} > $D/${arm}_r${k}.log 2>&1 || { echo "round $k arm $arm FAILED"; tail -n 5 $D/${arm}_r${k}.log; return; }
  f=$(ls -t runs/e18/latency/*_e253_${arm}_r${k}.json | head -1)
  python3 - "$f" "$D/${arm}_r${k}.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
json.dump(dict(arm=sys.argv[1].split("_e253_")[1][0], ms_per_frame=r["total"]["median_ms"], asm_ms=r["assembly"]["median_ms"], builder_ms=r["builder"]["median_ms"],
               points=r["points_per_frame_mean"], src=sys.argv[1]), open(sys.argv[2], "w"))
PYEOF
  echo "round $k arm $arm: $(cat $D/${arm}_r${k}.json)"
}
for k in 1 2 3 4 5 6; do for arm in a c b d e f; do rep $arm $k; done; done
for k in 7 8; do for arm in a c b d e f; do [ "$(nvalid $arm)" -lt 6 ] && rep $arm $k; done; done
echo "E253 replicates done"
