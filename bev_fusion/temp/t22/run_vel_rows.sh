#!/bin/bash
# scratch: prints the rows of vel_rows.py for the pattern in $1 (limit $2)
cd /mnt/c/Auto-research/bev_fusion || exit 1
python3 temp/t22/vel_rows.py "$@"
