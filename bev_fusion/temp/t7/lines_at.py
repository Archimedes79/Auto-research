"""t7 scratch: mean training group.lines over steps 900-1000 of runs (kill criterion): python temp/t7/lines_at.py <run> ..."""
import json
import sys

for r in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"runs/{r}/log.jsonl") if l.startswith('{"step"')]
    v = [x["group.lines"] for x in rows if "group.lines" in x and "val" not in x and 900 <= x["step"] <= 1000]
    print(r, f"group.lines mean over steps 900-1000: {sum(v) / max(len(v), 1):.3f} ({len(v)} log rows)")
