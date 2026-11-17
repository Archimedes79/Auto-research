#!/bin/bash
# scratch of the E357 build: run one project script when the 1-minute load is at most 14; output to temp/e357/<name>.log, the last lines shown
#   wsl bash temp/e357/e357_run_tmp.sh <name> <script.py> [args...]
name="$1"; shift
date; uptime
l=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$l" 'BEGIN{exit !(l > 14)}'; then echo LOAD_TOO_HIGH; exit 3; fi
SECONDS=0
bash tools/env/_py14.sh "$@" > "temp/e357/$name.log" 2>&1
echo "exit $? after $SECONDS s"
tail -n 45 "temp/e357/$name.log"
