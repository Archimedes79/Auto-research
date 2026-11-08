#!/bin/bash
# is any check of test_host_overhead flaky? run it 8 times, with 2 and with 6 threads
cd /mnt/c/Auto-research/bev_fusion || exit 1
for n in 2 6; do
  for i in 1 2 3 4; do
    OMP_NUM_THREADS=$n ~/miniconda3/envs/gcar/bin/python tests/test_host_overhead.py 2>&1 | grep -E "FAIL|GATE" | tr '\n' ' '
    echo " (threads $n, run $i)"
  done
done
