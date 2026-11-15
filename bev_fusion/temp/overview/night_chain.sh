#!/bin/bash
# waits for the t20 read-out to finish, then runs the cached-map inference + read-out of the two fine-tune arms, one after the other (CPU only, nice 15, 2 threads)
cd /mnt/c/Auto-research/bev_fusion
until grep -q "^-- 5" temp/overview/lr_t20.log 2>/dev/null || grep -q "Traceback" temp/overview/lr_t20.log 2>/dev/null; do sleep 10; done
for r in t21_ctl_s0 t23_C_s0; do
  bash temp/overview/rp_py.sh lane_readout.py run $r > temp/overview/lr_$r.log 2>&1
  echo "$(date +%H:%M) lane_readout $r finished (exit $?)" >> temp/overview/night_chain.log
done
echo done >> temp/overview/night_chain.log
