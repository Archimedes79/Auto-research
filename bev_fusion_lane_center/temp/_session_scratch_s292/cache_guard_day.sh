#!/bin/bash
# WSL ENOMEM guard for the night (OSError 12 on /mnt/c gzip reads when the page cache fills the VM: E355.s25 died of it at 01:11): every 60 s, when buff/cache is above 15 GB, drop the page cache (as root; nothing is lost, reads are only slower for a moment). Ends at 17:30 (D111 shutdown).
LOG=/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_s292/cache_guard_day.log
END=$(date -d "today 17:30" +%s)
[ "$(date +%s)" -gt "$END" ] && END=$(date -d "tomorrow 17:30" +%s)
while [ "$(date +%s)" -lt "$END" ]; do
  c=$(free -m | awk '/^Mem:/{print $6}')
  if [ "${c:-0}" -gt 15000 ]; then
    sync
    echo 1 > /proc/sys/vm/drop_caches && echo "$(date +%H:%M:%S) dropped (buff/cache was ${c} MB)" >> $LOG
  fi
  sleep 60
done
echo "$(date +%H:%M:%S) guard ended" >> $LOG
