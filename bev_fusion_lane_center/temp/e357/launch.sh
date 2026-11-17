#!/bin/bash
# E357: start the night chain detached from the calling shell (a chain started from a chat's shell dies with it).
cd "$(dirname "$0")/../.."
setsid nohup bash temp/e357/chain.sh > temp/e357/chain_stdout.log 2>&1 < /dev/null &
sleep 20
tail -n 3 temp/e357/chain.log
ps -eo pid,etimes,cmd | grep -E "e357/chain.sh|run.py --id E357" | grep -v grep | cut -c1-110
