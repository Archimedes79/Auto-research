#!/bin/bash
# scratch: what runs on the box now (GPU jobs of run.py, memory), the schedule view of t23 / t22, hold markers
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
date
echo "--- run.py / training processes"
ps -eo pid,etime,args | grep -E 'run\.py|train\.py|run_ar|qgraph_train|evaluate\.py|chain4|bev_suite' | grep -v grep | cut -c1-190
echo "--- gpu"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | head
echo "--- schedule show"
python3 ../autoresearch/tools/schedule.py show 2>&1 | cut -c1-220 | head -40
echo "--- queue"
ls temp/lane_tests/ 2>/dev/null | head -20
