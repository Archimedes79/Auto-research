#!/bin/bash
cd "$(dirname "$0")/../.."
setsid nohup bash temp/e359/chain.sh > temp/e359/chain_stdout.log 2>&1 < /dev/null &
sleep 3
tail -n 2 temp/e359/chain.log
ps -eo pid,etimes,cmd | grep -E "[e]359/chain.sh" | cut -c1-80
