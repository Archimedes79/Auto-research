"""throw-away: why does the step time rise during t18.fix / t20.base_s0 (3.2 -> 4.1 s) but not during t17.data? columns of the log that move with the step time"""
import json, statistics as st, sys
def blocks(run, keys):
    rows = [json.loads(l) for l in open(f"runs/{run}/log.jsonl")]
    tr = [r for r in rows if "s_per_step" in r and "t_data" in r]
    print(run)
    print("   block      s/step " + " ".join(f"{k[:14]:>14s}" for k in keys))
    for a in range(0, 6000, 1000):
        b = [r for r in tr if a < r["step"] <= a + 1000]
        vals = []
        for k in keys:
            v = [r[k] for r in b if isinstance(r.get(k), (int, float))]
            vals.append(st.median(v) if v else float("nan"))
        print(f"   {a+1:5d}-{a+1000:5d} {st.median(r['s_per_step'] for r in b):6.2f} " + " ".join(f"{v:14.3f}" for v in vals))
keys = ["gpu_gb", "radar_kept", "radar_aux", "total", "group.objects", "objhead.vel@+0.0"]
for run in ["t17_data", "t18_fix", "t20_base_s0"]:
    blocks(run, keys)
rows = [json.loads(l) for l in open("runs/t20_base_s0/log.jsonl")]
print("keys of a t20 row:", sorted(k for k in rows[100] if not k.startswith(("upd.", "grad.", "s.", "group.", "objhead", "motion", "lane", "sem")))[:40])
