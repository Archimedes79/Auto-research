#!/bin/bash
# the selftest three times in a row: it must not be flaky
cd /mnt/c/Auto-research/autoresearch || exit 1
for k in 1 2 3; do
  python3 -m jobrunner selftest > /tmp/jr_rep$k.txt 2>&1; echo "run $k: rc=$? $(tail -n 1 /tmp/jr_rep$k.txt | cut -c1-120)"
done
