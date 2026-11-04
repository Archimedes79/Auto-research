#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion/temp/speed_cpu || exit 1
for spec in "1 15 s_b1n15" "2 15 s_b2n15" "1 30 s_b1n30"; do
  set -- $spec
  /usr/bin/time -v bash run.sh profile_step.py $1 $2 $3 small > out_$3.txt 2> err_$3.txt
  echo "done $3"; grep Maximum err_$3.txt
done
uptime
