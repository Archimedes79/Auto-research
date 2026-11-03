#!/bin/bash
# scratch: stop MY CPU rendering jobs (video render and the gallery that waits behind it) because they slow the t18.fix training by about 25 %;
# the renders move into the GPU window between t18.fix.eval and the control. Pids are looked up by script name, never by a pattern that matches this shell.
for pid in $(ps -eo pid,args | grep "[g]allery_cpu.sh" | awk '{print $1}'); do echo "kill gallery $pid"; kill "$pid"; done
sleep 1
for pid in $(ps -eo pid,args | grep "[b]ev_suite.py runs/t16_full_s0 --device cpu" | awk '{print $1}'); do echo "kill render $pid"; kill "$pid"; done
for pid in $(ps -eo pid,args | grep "[v]ideos_cpu_baseline.sh" | awk '{print $1}'); do echo "kill wrapper $pid"; kill "$pid"; done
sleep 3
ps -eo pid,etimes,pcpu,args | grep -E "bev_suite|gallery|videos_cpu" | grep -v grep | cut -c1-140
echo done
