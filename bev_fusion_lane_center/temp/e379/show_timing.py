"""print one timing json of the E2E cost table (scratch, 2026-11-18)  usage: python show_timing.py <json>"""
import json, sys
d = json.load(open(sys.argv[1]))
print(sys.argv[1].split("/")[-1])
print(" host:", d.get("host")); print(" quiet", d.get("quiet"), "usable", d.get("usable_for_gcost"), "device", d.get("device"), "frames", d.get("frames"), "repeats", d.get("repeats"))
for k in ("median_ms", "p90_ms"):
    if k in d: print(" ", k, {a: round(b, 2) for a, b in d[k].items()})
for k in ("params", "params_whole_path", "stacked_max_diff"):
    if k in d: print(" ", k, d[k])
if "arms" in d:
    for k, v in d["arms"].items(): print("  arm", k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
    print("  block_cost_ms", d.get("block_cost_ms"))
