#!/bin/bash
dmesg 2>&1 | grep -iE "allocation failure|order:|oom|out of memory|Killed" | tail -8
echo ---
grep -E "MemFree|MemAvailable|^Cached|Dirty|Writeback:|Slab|Unevictable|Mlocked" /proc/meminfo
echo ---
cat /proc/sys/vm/min_free_kbytes /proc/sys/vm/overcommit_memory
sudo -n true 2>&1 | head -1
