#!/usr/bin/env bash
# Wait until no whole-chain training round runs (no process with "t1_r" in its command line; 3 checks 30 s apart, at most
# 12 h), then wait for a free GPU (tests/run_when_gpu_free.sh) and run the given command. Never touches another job.
# Usage: bash tests/run_after_t1.sh <command ...>
cd "$(dirname "$0")/.." || exit 1
free=0
for i in $(seq 1 1440); do
  if pgrep -f "[t]1_r" > /dev/null; then free=0; else free=$((free + 1)); fi
  [ "$free" -ge 3 ] && break
  sleep 30
done
[ "$free" -ge 3 ] || { echo "$(date +%T) t1 rounds still running after 12 h, not started"; exit 1; }
echo "$(date +%T) no t1 round runs: waiting for a free GPU"
exec bash tests/run_when_gpu_free.sh "$@"
