#!/bin/bash
# throw-away: one row per finished fused run (vehicles, moving, static, lanes, radial, tangential, boxes with / without
# radar, level 2 m / 1 m moving, IoU fwd / back, claims)
cd "$(dirname "$0")"
printf "%-26s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %s\n" run veh mov stat lane rad tan wRad noRad L2m L1m fwd back claims
for f in ${1:-f}*.log; do
  grep -q "claims hold" "$f" || continue
  v() { grep -m1 "$1" "$f" | awk '{print $NF}'; }
  printf "%-26s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %s\n" "${f%.log}" "$(v 'EPE vehicles  ')" \
    "$(v 'EPE vehicles moving')" "$(v 'EPE vehicles static')" "$(v 'EPE lanes')" "$(v 'radial part')" \
    "$(v 'tangential part')" "$(v 'WITH radar')" "$(v 'WITHOUT radar')" "$(v 'level 0')" "$(v 'level 1')" \
    "$(v 'IoU raw fwd v  ')" "$(v 'IoU raw back v  ')" "$(grep -o '[0-9] / 4' "$f")"
done
