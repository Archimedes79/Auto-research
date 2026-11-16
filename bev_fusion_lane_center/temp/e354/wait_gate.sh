#!/bin/bash
# E354: wait until the GPU has no training job (E353's short runs end about 18:35) and the host is quiet (1-min load < 4), at the latest 19:15,
# then run the K = 3 launch gate ALONE under a new name (the first one overlapped CPU tests) and print its clock line. Starts nothing else.
cd "$(dirname "$0")/../.."
busy() { ps -eo cmd | grep -E "run.py --id .*--gpu|jnarrow_train.py|jends_train.py|jmaps_train.py|jrecur_train.py" | grep -v grep | grep -q .; }
while :; do
  now=$(date +%H%M)
  load=$(awk '{print int($1)}' /proc/loadavg)
  if ! busy && [ "$load" -lt 4 ]; then break; fi
  if [ "$now" -ge 1915 ] && ! busy; then echo "host not quiet by 19:15 (load $load): gate runs anyway"; break; fi
  sleep 30
done
echo "gate starts $(date), load $(cat /proc/loadavg)"
bash temp/e354/gate.sh 3 4 e354_gate3 > temp/e354/gate3.out 2>&1
echo "gate exit $? $(date), load $(cat /proc/loadavg)"
grep -h "clock" runs/e354_gate3/train.log | cut -c1-300
ps -eo etimes,pcpu,cmd --sort=-pcpu | grep python | grep -v grep | head -5 | cut -c1-120
