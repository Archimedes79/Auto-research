#!/bin/bash
# scratch: wait (up to $2 x 10 s) for a line "exit N" in log $1, then show its tail
log="$1"; n="${2:-55}"
for ((i = 0; i < n; i++)); do
  grep -q '^exit' "$log" 2>/dev/null && break
  sleep 10
done
tail -14 "$log" | cut -c1-240
