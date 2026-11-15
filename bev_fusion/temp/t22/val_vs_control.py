"""scratch: training-time validation rows (same protocol: 50 val batches, EMA weights) of t22_B_s0 against t21_pair4_s0 at steps 750 and 1500"""
import json
def vals(run):
    out = {}
    for l in open(f"{run}/log.jsonl"):
        r = json.loads(l)
        if "val" in r:
            out[r["step"]] = r["val"]
    return out
a, b = vals("runs/t21_pair4_s0"), vals("runs/t22_B_s0")
keys = ["total", "group.areas", "group.lines", "group.objects", "group.motion", "motion@+0.0", "objhead.vel@+0.0", "objhead.mov@+0.0", "radar_kept", "total_last_warm", "total_last_cold"]
print("steps in control:", sorted(a), " in B:", sorted(b))
for st in (750, 1500):
    if st in a and st in b:
        print(f"step {st}:  key  control  B  B-control")
        for k in keys:
            if k in a[st] and k in b[st]:
                print(f"  {k:20s} {a[st][k]:9.4f} {b[st][k]:9.4f} {b[st][k]-a[st][k]:+9.4f}")
print("student_anchor in B:", {st: round(b[st].get("student_anchor", float('nan')), 4) for st in b})
