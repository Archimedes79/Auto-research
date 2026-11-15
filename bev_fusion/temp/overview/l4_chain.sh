#!/bin/bash
# JOB 4: the three variants of t20_base_s0 with the working tree's training/ code (CPU only, nice 15, 2 threads), one after the other
cd /mnt/c/Auto-research/bev_fusion
for v in live l4off nodepth; do
  bash temp/overview/rp_py.sh lane_readout.py run t20_base_s0 --variant $v > temp/overview/lr_t20_$v.log 2>&1
  echo "$(date +%H:%M) lane_readout t20_base_s0 --variant $v finished (exit $?)" >> temp/overview/l4_chain.log
done
echo done >> temp/overview/l4_chain.log
