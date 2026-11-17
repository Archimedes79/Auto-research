#!/bin/bash
cd "$(dirname "$0")/../.."
S=$(date +%s)
nice -n 19 bash tools/env/_py14.sh temp/e299/heat_replay.py --seed 0 --max 12 > temp/e299/smoke.txt 2>&1
echo "exit $? after $(( $(date +%s) - S )) s" >> temp/e299/smoke.txt
