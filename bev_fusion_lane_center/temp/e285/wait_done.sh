#!/bin/bash
# wait until the test list has printed DONE (max 20 min), then show it
cd "$(dirname "$0")"
for i in $(seq 1 1200); do
  grep -q "^DONE" gt_tests.txt && break
  sleep 1
done
cut -c1-180 gt_tests.txt
