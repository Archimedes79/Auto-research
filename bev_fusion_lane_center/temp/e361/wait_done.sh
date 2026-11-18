#!/bin/bash
# wait (at most 12 minutes) until the E361 chain3 has written its last line
cd "$(dirname "$0")/../.."
for i in $(seq 1 48); do
  if grep -q "chain3 done" temp/e361/chain.log; then break; fi
  sleep 15
done
date
tail -n 2 temp/e361/chain.log | cut -c1-120
