#!/bin/bash
date
for pat in "test_flow_explains" "test_flow_refine.py --quick$" "test_camera_only"; do
  pid=$(pgrep -f "[${pat:0:1}]${pat:1}" | head -1)
  [ -n "$pid" ] && echo "$pat pid $pid: $(ps -o stat=,etime=,time= -p $pid)  wchan $(cat /proc/$pid/wchan 2>/dev/null)"
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
