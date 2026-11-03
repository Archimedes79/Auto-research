#!/bin/bash
# waits until the trainer of run $1 is alive, then asks it to stop cleanly (trainsuite stop: checkpoint, STOPPED, exit 75, run.py puts the id on HOLD)
# the trainer moves an old CONTROL aside when it starts, so the request is sent only after the trainer runs
name=$1
cd "$(dirname "$0")/../../../autoresearch" || exit 1
for i in $(seq 1 120); do
  pgrep -f "runs/$name/code/train.py" > /dev/null && break
  sleep 3
done
pgrep -f "runs/$name/code/train.py" > /dev/null || { echo "$(date +%T) no trainer of $name started within 6 min: nothing to stop"; exit 0; }
echo "$(date +%T) trainer of $name is running"
sleep 25
python3 -m trainsuite --root ../bev_fusion/runs stop "$name" --reason "stopped at the user's request on 2026-11-01 (about 22:52): please stop all training activities; resume only on the user's word" 2>&1 | tail -6
for i in $(seq 1 60); do
  pgrep -f "runs/$name/code/train.py" > /dev/null || { echo "$(date +%T) trainer of $name has exited"; break; }
  sleep 5
done
python3 -m trainsuite --root ../bev_fusion/runs status "$name" 2>&1 | head -4
