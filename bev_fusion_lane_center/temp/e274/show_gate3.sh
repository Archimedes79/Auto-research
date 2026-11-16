#!/bin/bash
cd "$(dirname "$0")/../.."
for r in e274_r_s0 e274_r_s1 e274_p_s0 e274_p_s1; do
  echo "== $r"
  grep -h " GATE " runs/${r}_p*/log.txt | tail -1 | cut -d' ' -f3- > /tmp/g3.json
  python3 - <<PY
import json
j = json.load(open("/tmp/g3.json"))
for k, v in j.items():
    if k in ("p4", "p5", "p6", "p7") or not isinstance(v, dict):
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "flag_recall"} if isinstance(v, dict) else v)
PY
done
