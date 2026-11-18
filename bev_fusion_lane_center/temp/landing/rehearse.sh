#!/bin/bash
# rehearsal of the landing on the scratch copy: selftests of the files that changed (BFLC_ROOT = the copy)
cd "$(dirname "$0")/../.."
T=$PWD/temp/landing/tree/bev_fusion_lane_center
export BFLC_ROOT=$T OMP_NUM_THREADS=2
for s in "lane_past_data.py selftest" "token_time_ens.py selftest" "jtiming.py selftest" "lane_live_tok.py selftest"; do
  out=$(cd $T && nice -n 10 bash tools/env/_py14.sh $s 2>&1); echo "$? $s | $(echo "$out" | tail -1 | cut -c1-120)"
done
cd $T && python3 -m compileall -q training evaluation tools > /tmp/compile.out 2>&1; echo "compileall exit $? $(wc -l < /tmp/compile.out) lines"
