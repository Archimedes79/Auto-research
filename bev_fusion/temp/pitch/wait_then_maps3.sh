#!/bin/bash
# Like-for-like map scores of the case (GaussianCaR vs our chains on the same 400 frames: CPU, 2 threads, nice, NO loader workers, ONE process at a time), then the before / after frames.
# Starts when the box has room (free memory >= 9 GB and load < 8 for two minutes in a row), whatever trains or evaluates. A watchdog stops the job AND ITS CHILDREN (own process group)
# when free memory falls below 1.5 GB: the trainings and evaluations of the other chats come first (agreed with the queue owner, 2026-11-01 02:30). v3: v2 left orphaned loader workers behind.
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
avail() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
load1() { awk '{print int($1)}' /proc/loadavg; }
quiet() { [ "$(avail)" -ge "$1" ] && [ "$(load1)" -lt 8 ]; }
wait_quiet() {                       # $1 = free memory needed in MB; two quiet minutes in a row
  ok=0
  while [ $ok -lt 2 ]; do
    if quiet "$1"; then ok=$((ok + 1)); else ok=0; fi
    sleep 60
  done
}
run_guarded() {
  setsid "$@" &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ "$(avail)" -lt 1500 ]; then
      echo "memory low ($(avail) MB): stopping process group $pid"
      kill -TERM -- -$pid 2>/dev/null; sleep 3; kill -KILL -- -$pid 2>/dev/null
      wait $pid 2>/dev/null; return 9
    fi
    sleep 10
  done
  wait $pid
}
for r in t10_dual_s0 t8_ctl_s0 t8_bevfeat_s0; do
  [ -f temp/soa_gap/map_ref_${r}_400.json ] && continue
  wait_quiet 9000
  echo "start $r $(date +%T)"
  OMP_NUM_THREADS=2 run_guarded nice -n 10 $PY -u temp/soa_gap/map_ref_w0.py runs/$r temp/soa_gap/map_ref_${r}_400.json 400 > temp/soa_gap/map_ref_${r}_400.log 2>&1
  echo "$r exit $? $(date +%T)"
done
echo "maps done $(date +%T)"
wait_quiet 9000
echo "start frames $(date +%T)"
OMP_NUM_THREADS=2 run_guarded nice -n 10 $PY -u temp/pitch/render_frames.py 6 temp/pitch/frames t8_ctl_s0 t10_dual_s0 > temp/pitch/render_frames.log 2>&1
echo "frames exit $? $(date +%T)"
