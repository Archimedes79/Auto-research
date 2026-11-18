#!/bin/bash
# Stop the E368 chain of the lane chat (close-out order of Mirko): supervisor first, then the run.py job.
# Patterns use [x] so that this script does not match itself.
sup=$(pgrep -f "[t]emp/e368/sup.sh")
echo "supervisor pids: $sup"
[ -n "$sup" ] && kill -TERM $sup
sleep 2
run=$(pgrep -f "[r]un.py --id E368")
echo "run.py pids: $run"
[ -n "$run" ] && kill -TERM $run
sleep 8
echo "left:"
ps -eo pid,etimes,cmd | grep -E "[r]un.py --id E368|[j]past_train|[s]up.sh" | cut -c1-120
ls -t /mnt/c/Auto-research/bev_fusion_lane_center/runs/e368_z_s0 2>/dev/null | head -5
tail -3 /mnt/c/Auto-research/bev_fusion_lane_center/runs/e368_z_s0/log.txt 2>/dev/null | cut -c1-200
