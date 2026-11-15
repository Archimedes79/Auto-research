#!/bin/bash
# scratch: shows one ledger row, and the usage of ledger.py note
cd /mnt/c/Auto-research/bev_fusion || exit 1
python3 temp/t22/show_row.py "$1"
echo ---
cd /mnt/c/Auto-research/autoresearch && AR_PROJECT=../bev_fusion python3 tools/ledger.py note --help 2>&1 | head -30
