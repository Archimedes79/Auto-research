#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
AR_PROJECT=../bev_fusion python3 -m trainsuite start --help 2>&1 | head -60 | cut -c1-230
