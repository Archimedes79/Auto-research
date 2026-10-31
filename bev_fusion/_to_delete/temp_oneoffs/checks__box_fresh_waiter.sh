#!/usr/bin/env bash
# Start the 4000-scene box optimality run only when the t1 training rounds are done: either the flag file temp/checks/box_fresh_go
# exists (the lead chat creates it), or it is 13:00 or later and no process with "t1_r" in its command line runs. Then it waits for
# a free GPU as tests/run_when_gpu_free.sh does. Checks every 60 s, at most 24 h.
for i in $(seq 1 1440); do
  if [ -e temp/checks/box_fresh_go ]; then break; fi
  if [ "$(date +%H%M)" -ge 1300 ] && ! pgrep -f "[t]1_r" > /dev/null; then break; fi
  sleep 60
done
echo "$(date +%T) t1 rounds done or flag set: waiting for a free GPU"
bash tests/run_when_gpu_free.sh /home/ml/miniconda3/envs/gcar/bin/python tests/test_box_optimality.py --configs 4000:6000
