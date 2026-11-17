#!/bin/bash
# wait (at most ~14 min) until the layer-1 log has its verdict line, then print it without the progress lines
LOG=${1:?log}
for i in $(seq 1 170); do grep -qE '^(PASS|FAIL)|Traceback' "$LOG" && break; sleep 5; done
grep -vE '^ +[0-9]+/' "$LOG" | head -60
