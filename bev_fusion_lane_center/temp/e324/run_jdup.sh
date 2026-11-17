#!/bin/bash
cd "$(dirname "$0")/../.."
for S in 0 1; do nice -n 19 bash tools/env/_py14.sh jdup.py score --seed $S >> temp/e324/jdup_s$S.log 2>&1; done
echo done > temp/e324/jdup.done
