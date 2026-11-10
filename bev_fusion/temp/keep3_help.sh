#!/bin/bash
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
python3 autoresearch/tools/keep3.py init --help 2>&1 | cut -c1-200
echo ----
python3 autoresearch/tools/keep3.py consider --help 2>&1 | cut -c1-200
echo ----
git -C /mnt/c/Auto-research status --short bev_fusion/records | head -5
grep -n "left" autoresearch/tools/keep3.py | head -12 | cut -c1-200
