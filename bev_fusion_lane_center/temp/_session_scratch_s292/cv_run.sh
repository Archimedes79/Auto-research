#!/bin/bash
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd $R
for S in 0 1; do nice bash tools/env/_py14.sh jwire_train.py cv --seed $S 2>&1 | tail -4 | cut -c1-260; done
