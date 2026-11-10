#!/bin/bash
# scratch: waits until the L4-off evaluation has ended, then starts the two CPU comparisons (one thread each, detached) and waits for them
#   L4 on vs L4 off          (what the lane branch adds to this checkpoint, under the live code)
#   stored t20_base_s0 vs L4 on  (does the live evaluator reproduce the stored numbers?)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
L=temp/t22/lane_off
n=0
while [ ! -f records/runs/diag.t20_L4off/done ] && [ $n -lt 90 ]; do sleep 20; n=$((n+1)); done
if [ ! -f records/runs/diag.t20_L4off/done ] || [ ! -f runs/diag_t20_L4off/eval/metrics.json ]; then echo "L4 off did not finish cleanly ($(date +%H:%M:%S))"; ls runs/diag_t20_L4off/eval 2>&1 | head -3; exit 1; fi
echo "L4 off done $(date +%H:%M:%S)"
setsid nohup $PY evaluation/bev_suite.py compare runs/diag_t20_L4on runs/diag_t20_L4off --curves --fusion --velocity --n 1000 > $L/compare_L4on_vs_off.log 2>&1 < /dev/null &
setsid nohup $PY evaluation/bev_suite.py compare runs/t20_base_s0 runs/diag_t20_L4on --curves --fusion --velocity --n 1000 > $L/compare_stored_vs_L4on.log 2>&1 < /dev/null &
sleep 5
while pgrep -f "bev_suite.py compare runs/(diag_t20_L4on|t20_base_s0) runs/diag_t20_L4" > /dev/null; do sleep 20; done
echo "L4 comparisons ended $(date +%H:%M:%S)"
wc -l $L/compare_L4on_vs_off.log $L/compare_stored_vs_L4on.log
