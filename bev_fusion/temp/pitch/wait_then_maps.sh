#!/bin/bash
# Wait until the box is quiet (no bev training running, enough free memory), then run the like-for-like map scores of the case
# (GaussianCaR vs our chains on the same 400 frames: CPU, 2 threads, ONE process at a time), then render the before / after frames.
# A watchdog stops the job when free memory falls below 1.5 GB: the trainings of the other chats come first (they were OOM-killed once).
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
avail() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
quiet() { [ "$(pgrep -f '[ /]train\.py' | wc -l)" -eq 0 ] && [ "$(avail)" -ge "$1" ]; }
wait_quiet() {                       # $1 = free memory needed in MB; three quiet minutes in a row
  ok=0
  while [ $ok -lt 3 ]; do
    if quiet "$1"; then ok=$((ok + 1)); else ok=0; fi
    sleep 60
  done
}
run_guarded() {
  "$@" &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ "$(avail)" -lt 1500 ]; then echo "memory low ($(avail) MB): stopping $pid"; kill $pid; wait $pid; return 9; fi
    sleep 10
  done
  wait $pid
}
for r in t10_dual_s0 t8_ctl_s0 t8_bevfeat_s0; do
  [ -f temp/soa_gap/map_ref_${r}_400.json ] && continue
  wait_quiet 9000
  echo "start $r $(date +%T)"
  OMP_NUM_THREADS=2 run_guarded $PY -u temp/soa_gap/map_ref.py runs/$r temp/soa_gap/map_ref_${r}_400.json 400 > temp/soa_gap/map_ref_${r}_400.log 2>&1
  echo "$r exit $? $(date +%T)"
done
echo "maps done $(date +%T)"
wait_quiet 9000
echo "start frames $(date +%T)"
OMP_NUM_THREADS=2 run_guarded $PY -u temp/pitch/render_frames.py 6 temp/pitch/frames t8_ctl_s0 t10_dual_s0 > temp/pitch/render_frames.log 2>&1
echo "frames exit $? $(date +%T)"
