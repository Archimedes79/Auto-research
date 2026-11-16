#!/bin/bash
# E357: start the ctl watcher detached.
cd "$(dirname "$0")/../.."
setsid nohup bash temp/e357/hold_ctl.sh > /dev/null 2>&1 < /dev/null &
sleep 3
cat temp/e357/hold_ctl.log
ps -eo pid,etimes,cmd | grep -E "[h]old_ctl.sh" | cut -c1-80
