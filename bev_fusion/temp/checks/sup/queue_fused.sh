#!/bin/bash
# GPU runs of tests/test_flow_fused.py, one at a time, on a SHARED GPU (CLAUDE.md: total < 11 GB): a run starts only while
# the GPU uses < 9.9 GB, is capped by torch at 0.1 of the card (GPU_FRACTION; it needs 0.8 GB), and is stopped by this
# queue (it is this queue's own child) if the card ever passes 11 GB - then retried when there is room again.
# Usage: bash queue_fused.sh <list file>: one run per line "<log name> <args...>"; CPU=1 runs on the CPU instead.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GPU_FRACTION=0.1
PY=~/miniconda3/envs/gcar/bin/python
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
free() { [ "$(used)" -lt 9900 ]; }
wait_free() { until free && sleep 30 && free; do sleep 60; done; }
while read -r name args <&3; do
  [ -z "$name" ] && continue
  [ -f "temp/checks/sup/$name.log" ] && grep -q "claims hold" "temp/checks/sup/$name.log" && continue   # done before
  while true; do
    if [ -n "$CPU" ]; then export CUDA_VISIBLE_DEVICES=; else wait_free; fi
    echo "$name start $(date +%T)"
    $PY -u tests/test_flow_fused.py $args < /dev/null > "temp/checks/sup/$name.log" 2>&1 &
    pid=$!
    cut=0
    while kill -0 $pid 2> /dev/null; do
      sleep 15
      if [ -z "$CPU" ] && [ "$(used)" -gt 11000 ]; then kill $pid; cut=1; fi
    done
    wait $pid
    rc=$?
    [ $cut -eq 0 ] && break
    echo "$name cut: GPU above 11 GB $(date +%T); retry when there is room"
    sleep 300
  done
  echo "$name exit $rc $(date +%T)"
done 3< "$1"
echo "queue finished $(date +%T)"
