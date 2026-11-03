#!/usr/bin/env bash
# events of the 6000-step run t10_dual_long_s0 (one line each): trainsuite watch (alerts, evals, the kill rule of the card) + a host-memory guard.
# Exits when the run is done, failed or stopped. Run it from anywhere inside WSL: wsl bash temp/t10/watch_long.sh
cd "$(dirname "$0")/../../../autoresearch" || exit 1
python3 -m trainsuite --root ../bev_fusion/runs watch t10_dual_long_s0 --kill "group.lines@1000 > 1.0" &      # card 3c item g (1.48 of the t7 rule was about twice the expected level of a 6000-step run)
wp=$!
last=0
while kill -0 $wp 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  now=$(date +%s)
  if [ "$avail" -lt 3500 ] && [ $((now - last)) -gt 600 ]; then echo "[$(date +%T)] LOW HOST MEMORY: ${avail} MB available"; last=$now; fi
  sleep 30
done
echo "[$(date +%T)] watch ended"
