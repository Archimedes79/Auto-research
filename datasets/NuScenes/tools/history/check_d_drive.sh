#!/usr/bin/env bash
# One-off (2026-10-25): can WSL write to D:, and how fast (one large file, many small files)?
set -euo pipefail
t=/mnt/d/datasets/NuScenes/_speedtest
mkdir -p "$t"
df -h /mnt/d | tail -1
s=$(date +%s.%N); dd if=/dev/zero of="$t/big.bin" bs=4M count=256 status=none; sync; e=$(date +%s.%N)
echo "1 GB write: $(echo "$e - $s" | bc) s"
s=$(date +%s.%N); cat "$t/big.bin" > /dev/null; e=$(date +%s.%N)
echo "1 GB read: $(echo "$e - $s" | bc) s"
s=$(date +%s.%N); for i in $(seq 1 500); do head -c 200000 /dev/urandom > "$t/f$i.jpg"; done; e=$(date +%s.%N)
echo "500 x 200 KB files write: $(echo "$e - $s" | bc) s"
s=$(date +%s.%N); cat "$t"/f*.jpg > /dev/null; e=$(date +%s.%N)
echo "500 x 200 KB files read: $(echo "$e - $s" | bc) s"
mkdir -p "$t/../_to_delete" && mv "$t" "$t/../_to_delete/speedtest_$(date +%H%M%S)"
