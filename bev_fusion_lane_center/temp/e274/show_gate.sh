#!/bin/bash
cd "$(dirname "$0")/../.."
for s in 0 1; do
  echo "== r_s$s"
  grep -h " GATE " runs/e274_r_s${s}_p*/log.txt | tail -1 | cut -d' ' -f3- > /tmp/g$s.json
  python3 - <<PY
import json
j = json.load(open("/tmp/g$s.json"))
for k, v in j.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
PY
done
