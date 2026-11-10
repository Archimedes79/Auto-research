#!/bin/bash
# prints available MB and the rss of the real gate a few times
for i in 1 2 3 4 5 6; do
  a=$(awk '/MemAvailable/ {print int($2/1000)}' /proc/meminfo)
  r=$(ps -eo rss,cmd | awk '/test_evaluation_api.py --real/ && !/awk/ {print int($1/1000)}' | head -1)
  echo "avail ${a} MB, gate rss ${r} MB"
  sleep 5
done
