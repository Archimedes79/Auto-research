#!/bin/bash
# E358 scratch (follow-up): the selftest of jwire_train, the staged tests and the mutation runs that changed, one after the other; log temp/e358/all_checks2.log
cd /mnt/c/Auto-research/bev_fusion_lane_center
L=temp/e358/all_checks2.log
: > $L
bash temp/e358/selftests.sh jwire_train jwire_data >> $L 2>&1
for f in qends_eval jwire_head jwire_data jwire_train; do
  bash temp/e358/run_scratch.sh temp/_session_scratch_e358/mutate.py $f > temp/e358/mut_$f.log 2>&1
  echo "== mutate $f: $(tail -1 temp/e358/mut_$f.log)" >> $L
  grep "SURVIVOR\|NOT APPLIED" temp/e358/mut_$f.log >> $L
done
echo ALL DONE >> $L
