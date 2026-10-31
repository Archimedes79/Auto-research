#!/usr/bin/env bash
# One-off: lower the CPU priority of this session's two background CPU jobs (the load average was 64 on 12 cores).
for pat in "[c]heck_fusion_binary.py" "[f]eature_cache.py build"; do
  for pid in $(pgrep -f "$pat"); do
    renice -n 12 -p "$pid" > /dev/null
    echo "pid $pid: $(ps -o ni=,pcpu=,etime= -p "$pid")"
  done
done
