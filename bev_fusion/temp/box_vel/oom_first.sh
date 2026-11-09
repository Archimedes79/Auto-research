#!/usr/bin/env bash
# t11 vel cause: mark this experiment's CPU side jobs (probe_features / extract_all) as the first victims of the OOM killer, so that on
# a full machine they die instead of another chat's training. Run: bash temp/box_vel/oom_first.sh
for p in $(pgrep -f 'temp/box_vel/(probe_features|extract_all)'); do
  echo 1000 > /proc/$p/oom_score_adj && echo "pid $p oom_score_adj $(cat /proc/$p/oom_score_adj)"
done
grep MemAvailable /proc/meminfo
