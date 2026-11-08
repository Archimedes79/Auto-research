#!/bin/bash
# scratch: wait until the killed control launcher's record is stale (> 306 s without a heartbeat), then relaunch the control behind t18.fix.eval and showcase.gpu
cd /mnt/c/Auto-research/bev_fusion || exit 1
HB=records/runs/t16.ctl_s0/heartbeat
while [ $(( $(date +%s) - $(stat -c %Y "$HB") )) -lt 307 ]; do sleep 2; done
echo "stale at $(date +%T)"
bash temp/t16/queue_showcase.sh b
