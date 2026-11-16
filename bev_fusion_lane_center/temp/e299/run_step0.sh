#!/bin/bash
cd "$(dirname "$0")/../.."
for S in 0 1; do
  nice -n 19 bash tools/env/_py14.sh jtime.py ceiling --seed $S > temp/e299/ceiling_s$S.log 2>&1
  nice -n 19 bash tools/env/_py14.sh jtime.py replay --seed $S --w 0.5 > temp/e299/replay_s$S.log 2>&1
done
echo done > temp/e299/step0.done
