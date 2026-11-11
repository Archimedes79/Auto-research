#!/bin/bash
# scratch: keep large free blocks in the WSL VM while E279 trains (the 9p reads of the DataLoader workers need order-4 pages; the page cache eats the free memory): when MemFree < 500 MB and the
# last reclaim is older than 10 minutes, touch 14 GB at user level and release it (temp/e276/reclaim.py). Runs until 07:00.
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
last=0
while [ "$(date +%H)" -lt 7 ]; do
  free_kb=$(awk '/MemFree/ {print $2}' /proc/meminfo)
  now=$(date +%s)
  if [ "$free_kb" -lt 500000 ] && [ $((now - last)) -ge 600 ]; then
    echo "$(date +%H:%M:%S) MemFree ${free_kb} kB: reclaim" >> temp/e276/mem_watch.log
    python3 temp/e276/reclaim.py > /dev/null 2>&1
    last=$(date +%s)
  fi
  sleep 60
done
