#!/bin/bash
# WSL: ENOMEM on 9p reads (E345.v_e324s_s0 13:23, E291.c_s0 13:29) while the page cache held 19.8 GB of the 24 GB VM and Windows had 1.3 GB free:
# drop the page cache (only caches, no process is touched) and compact memory. Run as root: wsl -u root bash <this file>
sync
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory
free -m | head -2
grep -E "^(MemFree|Cached|AnonPages)" /proc/meminfo
