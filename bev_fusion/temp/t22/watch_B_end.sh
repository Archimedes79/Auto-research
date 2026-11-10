#!/bin/bash
# scratch: the away window of the nights ends at 07:00; at 06:55 the trainer is asked to stop (JSON stop file, exit 75, a checkpoint is saved, resumable). Exits when the run has ended.
cd /mnt/c/Auto-research/bev_fusion || exit 1
while true; do
  if [ -f runs/t22_B_s0/.done ] || [ -f runs/t22_B_s0/STOPPED ]; then echo "run ended: $(ls runs/t22_B_s0 | grep -E '^(.done|STOPPED)$' | tr '\n' ' ')"; exit 0; fi
  now=$(date +%H%M)
  if [ "$now" -ge 0655 ] && [ "$now" -lt 0800 ]; then
    echo "{\"cmd\": \"stop\", \"reason\": \"t22 stage B: the night window closes at 07:00\", \"t\": $(date +%s)}" > runs/t22_B_s0/CONTROL
    echo "stop file written $(date +%T)"; exit 0
  fi
  sleep 60
done
