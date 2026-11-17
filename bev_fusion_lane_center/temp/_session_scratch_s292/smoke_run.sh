#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $R
nice bash tools/env/_py14.sh jwire_train.py run --seed 0 --name e325_smoke_s0 --max-train 200 --heads 1 --epochs 8 2>&1 | tail -9 | cut -c1-330
