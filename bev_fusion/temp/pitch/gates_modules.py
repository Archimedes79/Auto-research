"""add the two gate lines of the module probe / timing unit to a run_gates.sh (argv[1]): after the 'curve metrics' line"""
import sys

p = sys.argv[1]
lines = open(p, encoding="utf-8").read().split("\n")
new = ['run "module probe"      "$PY" evaluation/bev_modules.py --selftest      # tensor numbers, dead channels, planted velocity offsets, each response perturbation changes only what it names; no data',
       'run "module timing"     "$PY" evaluation/bev_timing.py --selftest       # toy modules with known sleeps: per call / per frame, calls, hooks removed, training split, idle rule; CPU']
if not any("bev_modules.py --selftest" in ln for ln in lines):
    i = next(k for k, ln in enumerate(lines) if ln.startswith('run "curve metrics"'))
    lines[i + 1:i + 1] = new
open(p, "w", encoding="utf-8").write("\n".join(lines))
print("ok")
