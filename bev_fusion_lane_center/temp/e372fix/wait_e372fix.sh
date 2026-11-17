#!/bin/bash
# throw-away: wait (at most ~18 min) until no process matches the pattern $1 (write it as [x]yz so that pgrep does not match itself)
for i in $(seq 1 110); do
  pgrep -f "$1" >/dev/null || break
  sleep 10
done
pgrep -f "$1" >/dev/null && echo still-running || echo finished
