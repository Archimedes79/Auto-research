#!/bin/bash
# scratch: where data/cache/img_front lives
cd /mnt/c/Auto-research/bev_fusion/data/cache || exit 1
ls -ld img_front
T=$(readlink -f img_front)
echo "target: $T"
df -h "$T" | tail -1
du -sh img_front/sweeps 2>/dev/null
