#!/usr/bin/env bash
# Append the maps-only job to the lane's queue (temp/study/queue3.txt, re-read by lane3.sh before every job). Idempotent.
cd "$(dirname "$0")/../study" || exit 1
Q=queue3.txt
JOB=cfg:t6_maps_only:0a3d73c
if grep -q -x -F "$JOB" $Q; then echo "already queued"; else
  [ -n "$(tail -c1 $Q)" ] && echo >> $Q
  echo "$JOB" >> $Q
  echo "queued"
fi
echo "--- queue now:"; cat $Q
echo "--- lane:"; tail -n 3 lane3b.log | cut -c1-160
