#!/bin/bash
# scratch of the t16 bring-up: one line per new validation row and per new trainsuite finding of a run, until it ends. usage: watch_run.sh <run folder name> [seconds between looks]
name=$1
every=${2:-300}
cd "$(dirname "$0")/../../../autoresearch" || exit 1
last_val=0
seen=""
while true; do
  log=../bev_fusion/runs/$name/log.jsonl
  if [ -f "$log" ]; then
    python3 - "$log" "$last_val" <<'PY' || true
import json, sys
log, last = sys.argv[1], int(sys.argv[2])
for ln in open(log):
    try:
        d = json.loads(ln)
    except ValueError:
        continue
    if "val" in d and d["step"] > last:
        v = d["val"]
        print("VAL step", d["step"], {k: round(x, 3) for k, x in v.items() if k in ("total", "group.areas", "group.lines", "group.objects", "group.motion", "radar_kept")}, flush=True)
PY
    last_val=$(python3 -c "
import json,sys
m=0
for ln in open('$log'):
    try: d=json.loads(ln)
    except ValueError: continue
    if 'val' in d: m=max(m,d['step'])
print(m)")
  fi
  out=$(python3 -m trainsuite --root ../bev_fusion/runs health "$name" 2>&1 | grep -E "alert|warn|ALERT|WARN|failed|stalled|nonfinite" | cut -c1-200)
  if [ -n "$out" ] && [ "$out" != "$seen" ]; then echo "HEALTH: $out"; seen="$out"; fi
  if [ -f ../bev_fusion/runs/$name/.done ]; then echo "DONE $name"; tail -n 3 ../bev_fusion/runs/$name/train.log | cut -c1-200; exit 0; fi
  if [ -f ../bev_fusion/runs/$name/STOPPED ]; then echo "STOPPED $name"; exit 0; fi
  if grep -q Traceback ../bev_fusion/runs/$name/train.log 2>/dev/null; then echo "FAILED $name"; tail -n 20 ../bev_fusion/runs/$name/train.log | cut -c1-200; exit 1; fi
  sleep "$every"
done
