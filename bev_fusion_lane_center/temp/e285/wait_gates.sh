#!/bin/bash
# wait until the second gate run has printed DONE (max 25 min), then show it
cd "$(dirname "$0")"
for i in $(seq 1 1500); do
  grep -q "^DONE" gates2.txt && break
  sleep 1
done
cut -c1-150 gates2.txt
