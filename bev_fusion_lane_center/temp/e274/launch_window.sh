#!/bin/bash
# E274: wait for Mirko's GPU window (Monday-Friday 08:30-17:30, every night 22:00-07:00) with enough time left for the job, the READY flag (critic pass done) and a free GPU, then start r_s0 and r_s1;
# the p jobs follow through launch_p.sh (they wait for their r run and for a window with 84 minutes left)
cd "$(dirname "$0")/../.."
now_min() { echo $((10#$(date +%H) * 60 + 10#$(date +%M))); }
window_left() {                                   # minutes left in the current window (0 = outside)
  d=$(date +%u); n=$(now_min)
  if [ $n -ge 1320 ]; then echo $((1440 - n + 420)); return; fi
  if [ $n -lt 420 ]; then echo $((420 - n)); return; fi
  if [ $d -le 5 ] && [ $n -ge 510 ] && [ $n -lt 1050 ]; then echo $((1050 - n)); return; fi
  echo 0
}
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2600 ] && ! pgrep -f "train.*\.py|evaluate\.py|qgraph_train|trainsuite" | grep -v $$ > /dev/null
}
NEED=${1:-165}                                     # minutes: r 2.4 h + margin
until [ -f temp/e274/READY ] && [ "$(window_left)" -ge "$NEED" ] && gpu_free; do sleep 30; done
echo "start $(date), window left $(window_left) min"
nohup bash temp/e274/start_arm.sh r 0 > temp/e274/start_r0.log 2>&1 &
sleep 20
nohup bash temp/e274/start_arm.sh r 1 > temp/e274/start_r1.log 2>&1 &
sleep 5
nohup bash temp/e274/launch_p.sh 0 > temp/e274/launch_p0.log 2>&1 &
nohup bash temp/e274/launch_p.sh 1 > temp/e274/launch_p1.log 2>&1 &
echo "launched $(date)"
