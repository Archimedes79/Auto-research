#!/bin/bash
# scratch: queue the idle-GPU timing of the three systems behind t18.fix.eval (a diagnostic run: --no-card, no verdict)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
nohup python3 /mnt/c/Auto-research/autoresearch/tools/run.py --id timing.idle3 --gpu --split val --no-card --wait-after 600 --after t18.fix.eval -- bash temp/t16/timing_all.sh > runs/timing_idle3.launcher.log 2>&1 &
sleep 4
ps -eo pid,etime,args | grep "[t]iming.idle3" | cut -c1-200
tail -n 3 runs/timing_idle3.launcher.log
