#!/bin/bash
cd "$(dirname "$0")/../.."
setsid nohup bash temp/e357/chain2.sh > temp/e357/chain2_stdout.log 2>&1 < /dev/null &
sleep 30
tail -n 3 temp/e357/chain.log
ps -eo pid,etimes,cmd | grep -E "[c]hain2.sh|run.py --id E357" | cut -c1-110
