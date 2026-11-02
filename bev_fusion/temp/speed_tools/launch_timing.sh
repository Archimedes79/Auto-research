#!/bin/bash
# NOT STARTED YET. The lead (2026-11-02): the GPU is busy until ~18:00 (t16.full_s0, its eval, t16.ctl_s0). Run this only after the lead announced that the control t16.ctl_s0 has started:
# three 70-step timing runs of the NEW code (pinned to commit f13e593, t10 config), one after the other behind the control, through run.py's queue (RAM gate 20 GB, FIFO line)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
prev=t16.ctl_s0
for x in d e f; do
  setsid nohup python3 ../autoresearch/tools/run.py --id speed.$x --no-card --no-git --gpu --ram-mb 20000 --split train --after $prev --wait-after 1500 --wait-gpu 1500 \
    -- bash temp/train/run_ar.sh temp/compose/speed_$x.json f13e593 > runs/speed_$x.launcher.log 2>&1 < /dev/null &
  prev=speed.$x
  sleep 2
done
pgrep -af "run.py --id speed" | cut -c1-140
# afterwards: python3 -m trainsuite --root ../bev_fusion/runs speed speed_d speed_e speed_f --skip 3   (from autoresearch/, AR_PROJECT=../bev_fusion), and compare with speed_a3 (old code, 2.66 s/step)
