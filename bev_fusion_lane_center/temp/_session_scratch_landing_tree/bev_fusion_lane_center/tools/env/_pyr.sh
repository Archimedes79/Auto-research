#!/bin/bash
# like _py14.sh, but repeats the run (up to 4 times) when it dies with a CUDA 'unknown error' (seen while a training shares the GPU): _pyr.sh script.py [args]
D="$(cd "$(dirname "$0")" && pwd)"
for i in 1 2 3 4; do
  bash $D/_py14.sh "$@" > /tmp/_pyr_$$.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then cat /tmp/_pyr_$$.log; rm -f /tmp/_pyr_$$.log; exit 0; fi
  if grep -q "CUDA error" /tmp/_pyr_$$.log; then echo "[retry $i after CUDA error]"; sleep 5; continue; fi
  cat /tmp/_pyr_$$.log; rm -f /tmp/_pyr_$$.log; exit $rc
done
cat /tmp/_pyr_$$.log; exit 1
