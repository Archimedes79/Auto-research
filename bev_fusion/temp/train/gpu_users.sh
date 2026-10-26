#!/usr/bin/env bash
# One-off: which python processes hold the GPU (an open /dev/dxg handle in WSL) and their CUDA_VISIBLE_DEVICES.
for p in $(pgrep -f python); do
  n=$(ls -l /proc/$p/fd 2>/dev/null | grep -c -E "dxg|nvidia")
  [ "$n" -gt 0 ] || continue
  cvd=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep CUDA_VISIBLE || echo "CUDA_VISIBLE_DEVICES unset")
  echo "$p gpu-handles=$n $cvd $(ps -o etime=,args= -p $p | cut -c1-120)"
done
uptime
