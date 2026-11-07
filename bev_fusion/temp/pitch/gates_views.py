import sys
p = sys.argv[1]
lines = open(p, encoding="utf-8").read().split("\n")
new = 'run "view helpers"       "$PY" evaluation/bev_selftest.py --only "bev_views helpers"   # error colours of boxes, faded history, radar layer, velocity hue, moving layer, loss heat; seconds'
if not any('--only "bev_views helpers"' in ln for ln in lines):
    i = next(k for k, ln in enumerate(lines) if "bev_errors.py --selftest" in ln)
    lines.insert(i + 1, new)
open(p, "w", encoding="utf-8").write("\n".join(lines))
