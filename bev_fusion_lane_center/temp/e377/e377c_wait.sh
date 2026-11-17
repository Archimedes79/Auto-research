#!/bin/bash
# E377 throw-away: wait until a test log ends (ok / failure line), then show its tail.  usage: bash temp/e377/e377c_wait.sh <log> <ok text>
for i in $(seq 1 55); do
  grep -q "$2\|Error\|failed" "$1" && break
  sleep 10
done
date +%T
cut -c1-300 "$1" | tail -6
