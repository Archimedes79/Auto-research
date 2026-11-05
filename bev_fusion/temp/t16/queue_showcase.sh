#!/bin/bash
# scratch: queue the GPU window of the big report (showcase.gpu) behind t18.fix, and move the control behind it (the control trainer needs the GPU memory alone).
# step 1 (this script, part a): showcase.gpu; step 2 (part b, run >= 6 minutes after part a): relaunch the control with --after t18.fix.eval,showcase.gpu
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
RUN=/mnt/c/Auto-research/autoresearch/tools/run.py
case "$1" in
  a)
    nohup python3 $RUN --id showcase.gpu --gpu --split val --no-card --wait-after 1800 --after t18.fix -- bash temp/t16/showcase_gpu.sh > runs/showcase_gpu.launcher.log 2>&1 &
    sleep 3
    echo "queued showcase.gpu pid $!"
    # stop the control's waiting launcher; its record turns stale after 5 minutes
    for pid in $(ps -eo pid,args | grep "[r]un.py --id t16.ctl_s0 --gpu" | awk '{print $1}'); do echo "kill control launcher $pid"; kill "$pid"; done
    ;;
  b)
    bash temp/t16/start_ctl.sh --go --run-arg=--wait-after --run-arg=1800 --run-arg=--after --run-arg=t18.fix.eval,showcase.gpu 2>&1 | grep -E 'started|refus|error|ended|stale|running' | cut -c1-220
    sleep 3
    ps -eo pid,etimes,args | grep "[r]un.py --id" | cut -c1-150
    ;;
esac
