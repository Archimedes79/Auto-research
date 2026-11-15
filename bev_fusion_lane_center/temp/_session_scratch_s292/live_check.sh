#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $R
nice bash tools/env/_py14.sh jwire_data.py live --seed 0 --split val_clean --max 24 --workers 0 2>&1 | tail -6 | cut -c1-300
