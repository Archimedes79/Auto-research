#!/bin/bash
# scratch: relaunch t17.data (pinned to the baseline's code), its evaluation and the control behind it, once the killed launchers' records are stale
age() { python3 -c "import os,time; print(int(time.time()-os.path.getmtime('/mnt/c/Auto-research/bev_fusion/records/runs/$1/heartbeat')))"; }
for id in t17.data t17.data.eval t16.ctl_s0; do until [ "$(age $id)" -gt 308 ]; do sleep 10; done; done
bash /mnt/c/Auto-research/bev_fusion/temp/t16/queue_t17b.sh
sleep 3
bash /mnt/c/Auto-research/bev_fusion/temp/t16/start_ctl.sh --go --run-arg=--after --run-arg=t17.data.eval 2>&1 | grep -E 'started|refus|error|ended' | cut -c1-200
sleep 2
ps -eo pid,etimes,cmd | grep 'run.py --id t1[67]' | grep -v grep | cut -c1-110
