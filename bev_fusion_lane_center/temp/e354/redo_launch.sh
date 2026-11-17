#!/bin/bash
cd "$(dirname "$0")/../.."
setsid nohup bash temp/e354/redo_flip.sh $1 > /dev/null 2>&1 < /dev/null &
sleep 2
ps -eo pid,etimes,cmd | grep -E "[r]edo_flip" | cut -c1-80
