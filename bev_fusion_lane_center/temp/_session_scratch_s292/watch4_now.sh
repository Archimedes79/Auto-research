#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
for p in $(pgrep -f "^bash temp/_session_scratch_s292/e351_quiet_watch4.sh"); do kill $p && echo "killed old watcher $p"; done
for p in $(pgrep -f "^bash temp/_session_scratch_lead/quiet_pause.sh"); do kill $p && echo "killed pause script $p (E352 is through, nothing to hold)"; done
sleep 1
(QUIET_NOW=1 nohup bash temp/_session_scratch_s292/e351_quiet_watch4.sh >> temp/_session_scratch_s292/e351_quiet_watch4.log 2>&1 &)
sleep 5
cat temp/_session_scratch_s292/e351_quiet_watch4.log | tail -3
ps -eo pid,etime,cmd | grep -E "quiet_watch4|run.py --id" | grep -v grep | cut -c1-100
