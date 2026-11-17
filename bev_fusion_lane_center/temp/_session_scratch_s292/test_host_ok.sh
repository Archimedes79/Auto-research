#!/bin/bash
eval "$(grep '^host_ok()' temp/e317/chain.sh)"
echo "epoch of expiry: $(date -d '2026-11-16 00:00' +%s), now: $(date +%s)"
R=8; Lm=8; [ "$(date +%s)" -lt "$(date -d '2026-11-16 00:00' +%s)" ] && { R=6; Lm=10; }
echo "limits now: RAM >= $R GB, load < $Lm"
free -g | sed -n 2p
cat /proc/loadavg
host_ok && echo "host_ok: yes" || echo "host_ok: no"
