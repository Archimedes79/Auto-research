#!/bin/bash
# E274: start arm p of one seed when its r run is done AND a GPU window has at least 84 minutes left (p ~1.1 h + margin) and no other GPU job runs.  usage: launch_p.sh SEED
cd "$(dirname "$0")/../.."
S=$1
now_min() { echo $((10#$(date +%H) * 60 + 10#$(date +%M))); }
window_left() {
  d=$(date +%u); n=$(now_min)
  if [ $n -ge 1320 ]; then echo $((1440 - n + 420)); return; fi
  if [ $n -lt 420 ]; then echo $((420 - n)); return; fi
  if [ $d -le 5 ] && [ $n -ge 510 ] && [ $n -lt 1050 ]; then echo $((1050 - n)); return; fi
  echo 0
}
gpu_free() {                                       # other chats' evaluation / queue jobs; the GPU slots and memory are run.py's own gate
  ! pgrep -f "evaluate\.py|trainsuite" | grep -v $$ > /dev/null
}
until [ -f experiment_execution/runs/E274.r_s$S/done ]; do sleep 60; done
until [ "$(window_left)" -ge 84 ] && gpu_free; do sleep 30; done
echo "p_s$S start $(date), window left $(window_left) min"
bash temp/e274/start_arm.sh p $S
