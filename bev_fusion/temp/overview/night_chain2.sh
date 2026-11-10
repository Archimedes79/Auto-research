#!/bin/bash
# the two fine-tune arms of job 2(d): cached-map inference + read-out + decode study (CPU only, nice 15, 2 threads), one after the other; finished frames of an earlier start are reused
cd /mnt/c/Auto-research/bev_fusion
for r in t21_ctl_s0 t23_C_s0; do
  bash temp/overview/rp_py.sh lane_readout.py run $r > temp/overview/lr_$r.log 2>&1
  echo "$(date +%H:%M) lane_readout $r finished (exit $?)" >> temp/overview/night_chain.log
  bash temp/overview/rp_py.sh lane_decode_study.py run $r > temp/overview/ds_$r.log 2>&1
  echo "$(date +%H:%M) lane_decode_study $r finished (exit $?)" >> temp/overview/night_chain.log
done
echo done >> temp/overview/night_chain.log
