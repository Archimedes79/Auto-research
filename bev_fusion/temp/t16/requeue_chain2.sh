#!/bin/bash
# scratch: move the control behind the bundle arm: kill its launcher first (done by the caller), wait for the stale record, queue t18.*, then the control after t18.fix.eval
age() { python3 -c "import os,time; print(int(time.time()-os.path.getmtime('/mnt/c/Auto-research/bev_fusion/records/runs/$1/heartbeat')))"; }
until [ "$(age t16.ctl_s0)" -gt 308 ]; do sleep 10; done
bash /mnt/c/Auto-research/bev_fusion/temp/t16/queue_t18.sh
sleep 3
bash /mnt/c/Auto-research/bev_fusion/temp/t16/start_ctl.sh --go --run-arg=--after --run-arg=t18.fix.eval 2>&1 | grep -E 'started|refus|error|ended' | cut -c1-200
sleep 2
ps -eo pid,etimes,cmd | grep 'run.py --id t1[678]' | grep -v grep | cut -c1-110
