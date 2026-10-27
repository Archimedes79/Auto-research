#!/bin/bash
# loop 20 start: the planned rows of loop 19 that never had a run record are closed as cancelled (they would show up as work of loop 20 otherwise)
cd "$(dirname "$0")/../.."
python3 ../autoresearch/tools/schedule.py show | grep '^planned' | sed 's/^planned ([0-9]*): //' | tr ',' '\n' | sed 's/ //g' | while read it; do
  [ -n "$it" ] && python3 ../autoresearch/tools/schedule.py set "$it" cancelled > /dev/null && echo "cancelled $it"
done
python3 ../autoresearch/tools/schedule.py show | cut -c1-160
