#!/bin/bash
# scratch: wait until run_all.sh is finished (at most 60 minutes)
for i in $(seq 1 360); do
  if ! pgrep -f "[d]eadcode/run_all.sh" > /dev/null; then echo "run_all finished"; exit 0; fi
  sleep 10
done
echo "timeout"; exit 1
