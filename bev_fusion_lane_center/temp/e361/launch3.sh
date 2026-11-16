#!/bin/bash
cd "$(dirname "$0")/../.."
setsid nohup bash temp/e361/chain3.sh > temp/e361/chain3_stdout.log 2>&1 < /dev/null &
sleep 3
tail -n 2 temp/e361/chain.log
ps -eo pid,etimes,cmd | grep -E "[c]hain3.sh" | cut -c1-70
