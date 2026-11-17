#!/bin/bash
# wait (throw-away) until a log holds one of the end markers, then print its tail: e369_wait.sh <log> <pattern> [seconds]
for i in $(seq 1 $(( ${3:-1800} / 10 ))); do
  grep -qE "$2" "$1" 2>/dev/null && break
  sleep 10
done
tail -6 "$1" | cut -c1-400
