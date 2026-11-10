#!/bin/bash
# scratch: wait until MemAvailable (MB) is above $1 (default 13000) for 3 polls in a row, at most 40 minutes
need=${1:-13000}
ok=0
for i in $(seq 1 240); do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -gt "$need" ]; then ok=$((ok+1)); else ok=0; fi
  if [ "$ok" -ge 3 ]; then echo "memory free: ${avail} MB"; exit 0; fi
  sleep 10
done
echo "timeout, last MemAvailable ${avail} MB"; exit 1
