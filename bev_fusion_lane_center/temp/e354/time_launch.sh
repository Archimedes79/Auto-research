#!/bin/bash
cd "$(dirname "$0")/../.."
setsid nohup bash temp/e354/time_chain.sh > temp/e354/time_chain_stdout.log 2>&1 < /dev/null &
sleep 3
tail -n 2 temp/e354/time_chain.log
ps -eo pid,etimes,cmd | grep -E "[t]ime_chain" | cut -c1-80
