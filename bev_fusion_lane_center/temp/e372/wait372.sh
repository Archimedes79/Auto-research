#!/bin/bash
# waits (read-only) until dumps372.sh writes its last line or 100 min pass; prints the chain log's tail
cd "$(dirname "$0")/../.."
for i in $(seq 1 200); do
  grep -q "\[dumps372\] all dumps and read-outs done" temp/e372/chain372.log && break
  sleep 30
done
tail -8 temp/e372/chain372.log
