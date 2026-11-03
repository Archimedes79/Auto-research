#!/bin/bash
# scratch: queue the idle-GPU timing of the three systems (a diagnostic run: --no-card, no verdict); second try after the first was refused by my own CPU job
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
ps -eo pid,args | grep -E "[b]ev_suite|[t]rain" | cut -c1-120
nohup python3 /mnt/c/Auto-research/autoresearch/tools/run.py --id timing.idle3b --gpu --split val --no-card -- bash temp/t16/timing_all.sh > runs/timing_idle3b.launcher.log 2>&1 &
sleep 4
ps -eo pid,etime,args | grep "[t]iming.idle3b" | cut -c1-200
