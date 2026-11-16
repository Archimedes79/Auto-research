#!/bin/bash
# run the selftest of every mutant (it must fail), one at a time, niced; the summary goes to temp/_session_scratch_e351/mut/summary.txt
cd /mnt/c/Auto-research/bev_fusion_lane_center
D=temp/_session_scratch_e351/mut
: > $D/summary.txt
while read -r name f; do
  s=$(date +%s)
  nice -n 10 bash tools/env/_py14.sh $D/$name/$f selftest > $D/$name/out.txt 2>&1
  rc=$?
  echo "$name rc=$rc $(( $(date +%s) - s )) s | $(tail -1 $D/$name/out.txt | cut -c1-200)" >> $D/summary.txt
done < $D/list.txt
echo DONE >> $D/summary.txt
