#!/usr/bin/env bash
# wait until at least $1 MB are available, then run the rest of the arguments (from the bev_fusion folder); up to 4 tries
# (the nuScenes table read over drvfs fails now and then with ENOMEM while the box is under memory pressure)
need="$1"; shift
cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for try in 1 2 3 4; do
  for i in $(seq 1 720); do
    avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -ge "$need" ]; then break; fi
    sleep 10
  done
  echo "try $try: available MB $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo), start $(date +%T)"
  start=$(date +%s)
  "$@" > /tmp/proto_try.log 2>&1
  rc=$?
  tail -25 /tmp/proto_try.log
  echo "try $try: exit $rc after $(( $(date +%s) - start )) s at $(date +%T)"
  if [ $rc -eq 0 ] || ! grep -q "Cannot allocate memory" /tmp/proto_try.log; then exit $rc; fi
  sleep 30
done
exit 1
