#!/bin/bash
cd "$(dirname "$0")/../.."
for r in e273_p_s0 e273_p_s1; do
  echo "== $r"
  grep -h " GATE " runs/${r}_p*/log.txt | tail -1 | cut -d' ' -f3- > /tmp/g.json
  python3 - <<PY
import json
j = json.load(open("/tmp/g.json"))
for k, v in j.items():
    if isinstance(v, dict):
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "flag_recall"})
    else:
        print(k, v)
PY
done
