#!/bin/bash
# wait until prepush.txt has DONE (max 15 min), then show it
cd "$(dirname "$0")"
for i in $(seq 1 900); do
  grep -q "^DONE" prepush.txt 2>/dev/null && break
  sleep 1
done
cut -c1-200 prepush.txt
