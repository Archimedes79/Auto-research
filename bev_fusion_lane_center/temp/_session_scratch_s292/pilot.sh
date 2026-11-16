#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $R
nice bash tools/env/_py14.sh jwire_data.py live --seed 0 --split train --max 200 --workers 2 2>&1 | tail -4 | cut -c1-400
