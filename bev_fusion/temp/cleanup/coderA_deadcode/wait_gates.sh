#!/bin/bash
# scratch: wait until my run_gates.sh is finished (at most 60 minutes)
for i in $(seq 1 360); do
  if ! pgrep -f "[r]un_gates.sh" > /dev/null; then echo "gates finished"; exit 0; fi
  sleep 10
done
echo "timeout"; exit 1
