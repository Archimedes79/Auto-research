#!/bin/bash
# E358 scratch: the four selftests, the staged tests and the four mutation runs, one after the other (CPU, nice, 2 threads); logs in temp/e358/all_checks.log
cd /mnt/c/Auto-research/bev_fusion_lane_center
L=temp/e358/all_checks.log
: > $L
bash temp/e358/selftests.sh >> $L 2>&1
bash temp/e358/run_scratch.sh temp/_session_scratch_e358/test_staged.py >> $L 2>&1
for f in jwire_data jwire_train jwire_verdict jwire_check; do
  bash temp/e358/run_scratch.sh temp/_session_scratch_e358/mutate.py $f > temp/e358/mut_$f.log 2>&1
  echo "== mutate $f: $(tail -1 temp/e358/mut_$f.log)" >> $L
  grep "SURVIVOR\|NOT APPLIED" temp/e358/mut_$f.log >> $L
done
echo ALL DONE >> $L
