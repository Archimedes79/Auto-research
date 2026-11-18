#!/bin/bash
# step 0c, both seeds, CPU only, niced, 2 threads; finished well before 21:00
cd "$(dirname "$0")/../.."
for s in 0 1; do
  S=$(date +%s)
  nice -n 19 bash tools/env/_py14.sh temp/e299/heat_replay.py --seed $s > temp/e299/full_s$s.txt 2>&1
  echo "exit $? after $(( $(date +%s) - S )) s" >> temp/e299/full_s$s.txt
done
echo ALLDONE > temp/e299/full_done.txt
