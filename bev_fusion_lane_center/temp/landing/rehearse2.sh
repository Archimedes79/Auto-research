#!/bin/bash
cd "$(dirname "$0")/../.."
T=$PWD/temp/_session_scratch_landing_tree/bev_fusion_lane_center
export BFLC_ROOT=$T OMP_NUM_THREADS=2
for s in "lane_ptok_data.py selftest" "lane_past_data.py selftest" "token_time_ens.py selftest" "jtiming.py selftest" "lane_live_tok.py selftest"; do out=$(cd $T && nice -n 10 bash tools/env/_py14.sh $s 2>&1); echo "$? $s | $(echo "$out" | tail -1 | cut -c1-120)"; done
