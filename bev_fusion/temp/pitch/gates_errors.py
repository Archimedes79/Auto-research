import sys
p = sys.argv[1]
lines = open(p, encoding="utf-8").read().split("\n")
new = 'run "error search"       "$PY" evaluation/bev_errors.py --selftest       # FP / FN kinds, counts equal the metric\'s matching, frame trace through the files; no data'
if not any("bev_errors.py --selftest" in ln for ln in lines):
    i = next(k for k, ln in enumerate(lines) if "bev_timing.py --selftest" in ln)
    lines.insert(i + 1, new)
open(p, "w", encoding="utf-8").write("\n".join(lines))
