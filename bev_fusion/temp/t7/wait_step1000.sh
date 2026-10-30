#!/usr/bin/env bash
# t7 kill check: waits until a run's log reaches step 1000 (or the run ends), then prints its group.lines at steps 900-1000.
# Usage: bash temp/t7/wait_step1000.sh <run name>
cd "$(dirname "$0")/../.." || exit 1
r=$1
until grep -q '"step": 1000,' runs/$r/log.jsonl 2>/dev/null || [ -f runs/$r/.done ] || grep -q "cfg:$r:.*no .done" temp/study/lane3b.log 2>/dev/null; do
  sleep 120
done
python3 temp/t7/lines_at.py $r
echo "kill threshold 1.48 (1.2 x the reference's 1.235)"
