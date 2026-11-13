#!/bin/bash
# wait until the three pre-push tests have written their EXIT line (max 15 min), then show the last lines of each
cd "$(dirname "$0")"
for i in $(seq 1 900); do
  n=0
  for f in qtargets2.txt last_jmaps.py.txt last_test_jmaps_dirx.py.txt; do
    grep -q "^EXIT=" "$f" 2>/dev/null && n=$((n+1))
  done
  [ $n -eq 3 ] && break
  sleep 1
done
for f in qtargets2.txt last_jmaps.py.txt last_test_jmaps_dirx.py.txt; do echo "== $f"; tail -3 "$f" | cut -c1-220; done
